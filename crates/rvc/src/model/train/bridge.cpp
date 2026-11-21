#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/custom_function.h>

#include <cstdio>
#include <memory>

namespace {

constexpr auto cuda = at::kCUDA;

int autocast_state() {
  return (static_cast<int>(at::autocast::get_autocast_dtype(cuda)) << 1) |
      at::autocast::is_autocast_enabled(cuda);
}

void set_autocast(int state) {
  at::autocast::set_autocast_dtype(cuda, static_cast<at::ScalarType>(state >> 1));
  at::autocast::set_autocast_enabled(cuda, (state & 1) != 0);
}

using Run = at::Tensor* (*)(void*, const at::Tensor* const*, size_t, char*, size_t);
using Drop = void (*)(void*);

struct Invocation {
  std::shared_ptr<void> context;
  Run run;
  int amp;

  at::Tensor call(at::TensorList inputs) const {
    std::vector<const at::Tensor*> pointers;
    pointers.reserve(inputs.size());
    for (const auto& input : inputs) {
      pointers.push_back(&input);
    }
    char error[1024] = {};
    std::unique_ptr<at::Tensor> output(
        run(context.get(), pointers.data(), pointers.size(), error, sizeof(error)));
    TORCH_CHECK(output, "decoder checkpoint: ", error);
    return std::move(*output);
  }
};

struct ScopedAutocast {
  int previous = autocast_state();
  bool cache = at::autocast::is_autocast_cache_enabled();

  explicit ScopedAutocast(int state) {
    set_autocast(state);
    // Recomputed weights must have their own autograd history. Do not reuse
    // casts from the forward pass, which ran with autograd disabled.
    at::autocast::set_autocast_cache_enabled(false);
  }

  ~ScopedAutocast() {
    at::autocast::set_autocast_cache_enabled(cache);
    set_autocast(previous);
  }
};

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

class Checkpoint : public torch::autograd::Function<Checkpoint> {
 public:
  static at::Tensor forward(
      AutogradContext* ctx, at::Tensor owner, at::TensorList inputs) {
    ctx->saved_data["owner"] = owner;
    ctx->save_for_backward(inputs.vec());
    const auto& invocation = *static_cast<const Invocation*>(owner.const_data_ptr());
    ScopedAutocast amp(invocation.amp);
    return invocation.call(inputs);
  }

  static variable_list backward(AutogradContext* ctx, variable_list gradients) {
    auto saved = ctx->get_saved_variables();
    auto owner = ctx->saved_data.at("owner").toTensor();
    const auto& invocation = *static_cast<const Invocation*>(owner.const_data_ptr());
    at::AutoGradMode enable_grad(true);
    variable_list differentiable;
    std::vector<size_t> slots;
    for (size_t i = 0; i < saved.size(); ++i) {
      const bool requires_grad = saved[i].requires_grad();
      saved[i] = saved[i].detach().set_requires_grad(requires_grad);
      if (requires_grad) {
        differentiable.push_back(saved[i]);
        slots.push_back(i + 1);  // The first argument owns the callback.
      }
    }
    auto result = [&] {
      ScopedAutocast amp(invocation.amp);
      return invocation.call(saved);
    }();
    auto computed = torch::autograd::grad(
        {result}, differentiable, {gradients.at(0)}, false, false, true);
    variable_list output(saved.size() + 1);
    for (size_t i = 0; i < slots.size(); ++i) {
      output[slots[i]] = std::move(computed[i]);
    }
    return output;
  }
};

} // namespace

extern "C" int rvc_enter_autocast(int precision) {
  int previous = autocast_state();
  at::autocast::increment_nesting();
  at::autocast::set_autocast_enabled(cuda, precision != 0);
  if (precision != 0) {
    at::autocast::set_autocast_dtype(cuda, precision == 2 ? at::kBFloat16 : at::kHalf);
  }
  return previous;
}

extern "C" void rvc_leave_autocast(int previous) {
  if (at::autocast::decrement_nesting() == 0) {
    at::autocast::clear_cache();
  }
  set_autocast(previous);
}

extern "C" at::Tensor* rvc_checkpoint(
    void* context, Run run, Drop drop, const at::Tensor* const* inputs, size_t count,
    char* error, size_t error_length) {
  try {
    // shared_ptr invokes the drop function even if construction fails.
    auto lifetime = std::shared_ptr<void>(context, drop);
    auto invocation = std::make_shared<Invocation>(
        Invocation{std::move(lifetime), run, autocast_state()});
    auto owner = at::from_blob(invocation.get(), {1},
        [invocation](void*) {}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    variable_list arguments;
    arguments.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      arguments.push_back(*inputs[i]);
    }
    return new at::Tensor(Checkpoint::apply(owner, at::TensorList(arguments)));
  } catch (const c10::Error& exception) {
    std::snprintf(error, error_length, "%s", exception.what_without_backtrace());
  } catch (const std::exception& exception) {
    std::snprintf(error, error_length, "%s", exception.what());
  }
  return nullptr;
}

namespace {

std::vector<at::Tensor> tensor_list(at::Tensor* const* tensors, size_t count) {
  std::vector<at::Tensor> list;
  list.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    list.push_back(*tensors[i]);
  }
  return list;
}

void report(const char* what, char* error, size_t error_length) {
  std::snprintf(error, error_length, "%s", what);
}

} // namespace

// One multi-tensor AdamW update (torch.optim.AdamW(fused=True) semantics).
// All lists hold `count` tensors on one device with one floating dtype.
// The per-parameter step counters are incremented here, as in PyTorch.
extern "C" int rvc_fused_adamw(
    at::Tensor* const* params, at::Tensor* const* grads, at::Tensor* const* exp_avgs,
    at::Tensor* const* exp_avg_sqs, at::Tensor* const* steps, size_t count, double lr,
    double beta1, double beta2, double weight_decay, double eps, char* error,
    size_t error_length) {
  try {
    at::AutoGradMode no_grad(false);
    auto p = tensor_list(params, count);
    auto g = tensor_list(grads, count);
    auto m = tensor_list(exp_avgs, count);
    auto v = tensor_list(exp_avg_sqs, count);
    auto s = tensor_list(steps, count);
    at::_foreach_add_(s, 1);
    at::_fused_adamw_(p, g, m, v, {}, s, lr, beta1, beta2, weight_decay, eps, false, false);
    return 1;
  } catch (const c10::Error& exception) {
    report(exception.what_without_backtrace(), error, error_length);
  } catch (const std::exception& exception) {
    report(exception.what(), error, error_length);
  }
  return 0;
}

// Unscale gradients in place and flag non-finite values with one multi-tensor kernel.
extern "C" int rvc_foreach_unscale(
    at::Tensor* const* grads, size_t count, at::Tensor* found_inf, const at::Tensor* inv_scale,
    char* error, size_t error_length) {
  try {
    at::AutoGradMode no_grad(false);
    auto g = tensor_list(grads, count);
    at::_amp_foreach_non_finite_check_and_unscale_(g, *found_inf, *inv_scale);
    return 1;
  } catch (const c10::Error& exception) {
    report(exception.what_without_backtrace(), error, error_length);
  } catch (const std::exception& exception) {
    report(exception.what(), error, error_length);
  }
  return 0;
}

// Page-locked copy of a host tensor for asynchronous host-to-device transfers.
extern "C" at::Tensor* rvc_pin_memory(const at::Tensor* tensor, char* error, size_t error_length) {
  try {
    return new at::Tensor(tensor->pin_memory());
  } catch (const c10::Error& exception) {
    report(exception.what_without_backtrace(), error, error_length);
  } catch (const std::exception& exception) {
    report(exception.what(), error, error_length);
  }
  return nullptr;
}

// self[i] += weight * (end[i] - self[i]) for every tensor, as one multi-tensor kernel.
extern "C" int rvc_foreach_lerp(
    at::Tensor* const* self, at::Tensor* const* end, size_t count, double weight, char* error,
    size_t error_length) {
  try {
    at::AutoGradMode no_grad(false);
    auto a = tensor_list(self, count);
    auto b = tensor_list(end, count);
    at::_foreach_lerp_(a, b, weight);
    return 1;
  } catch (const c10::Error& exception) {
    report(exception.what_without_backtrace(), error, error_length);
  } catch (const std::exception& exception) {
    report(exception.what(), error, error_length);
  }
  return 0;
}
