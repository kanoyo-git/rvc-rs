#include <ATen/core/Tensor.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/ops/_scaled_dot_product_cudnn_attention.h>
#include <ATen/ops/_scaled_dot_product_efficient_attention.h>
#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <c10/util/Exception.h>
#include <exception>
#include <string>
#include <string_view>

// These LibTorch 2.7 exports are declared in cuda/sdp_utils.h in the source
// tree, but that header is omitted from the prebuilt distribution.
namespace sdp {
C10_IMPORT bool can_use_cudnn_attention(const sdp_params&, bool);
C10_IMPORT bool can_use_flash_attention(const sdp_params&, bool);
C10_IMPORT bool can_use_mem_efficient_attention(const sdp_params&, bool);
}

namespace {
const char* report_error(const char* error) noexcept {
    thread_local std::string message;
    try {
        message = error;
        return message.c_str();
    } catch (...) {
        return "failed to report the CUDA attention error";
    }
}
}

extern "C" const char* rvc_roformer_cuda_sdpa(
    const at::Tensor* query, const at::Tensor* key, const at::Tensor* value,
    int kernel, at::Tensor** output, int* no_cudnn_plan) noexcept {
    *output = nullptr;
    *no_cudnn_plan = 0;
    try {
        const sdp::sdp_params params{*query, *key, *value, std::nullopt, 0.0, false, false};
        // Dispatch a single kernel without changing LibTorch's process-wide
        // preferences. A null result lets the caller try the regular selector.
        switch (kernel) {
        case 0:
            if (sdp::can_use_cudnn_attention(params, false)) {
                *output = new at::Tensor(std::get<0>(at::_scaled_dot_product_cudnn_attention(
                    *query, *key, *value, std::nullopt, false, 0.0, false, false)));
            }
            break;
        case 1:
            // The public flash wrapper pads unaligned heads before dispatch.
            if (query->size(-1) % 8 == 0 && sdp::can_use_flash_attention(params, false)) {
                *output = new at::Tensor(std::get<0>(at::_scaled_dot_product_flash_attention(
                    *query, *key, *value, 0.0, false, false)));
            }
            break;
        case 2:
            if (sdp::can_use_mem_efficient_attention(params, false)) {
                *output = new at::Tensor(std::get<0>(at::_scaled_dot_product_efficient_attention(
                    *query, *key, *value, std::nullopt, false, 0.0, false)));
            }
            break;
        }
        return nullptr;
    } catch (const c10::OutOfMemoryError& error) {
        return report_error(error.what());
    } catch (const c10::Error& error) {
        const std::string_view message = error.what_without_backtrace();
        if (kernel == 0 && message.find("cuDNN Frontend error: No valid engine configs for ") == 0) {
            *no_cudnn_plan = 1;
            return nullptr;
        }
        return report_error(error.what());
    } catch (const std::exception& error) {
        return report_error(error.what());
    } catch (...) {
        return "unknown CUDA attention error";
    }
}
