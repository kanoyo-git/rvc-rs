#include <ATen/Context.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <thread>

namespace {

using c10::cuda::CUDAStream;
namespace allocator = c10::cuda::CUDACachingAllocator;

void report(const char* operation, const char* message, char* error, size_t capacity) noexcept {
    if (error != nullptr && capacity > 0) {
        std::snprintf(error, capacity, "%s", message);
    } else {
        std::fprintf(stderr, "RVC CUDA graph %s: %s\n", operation, message);
    }
}

void clear_error(char* error, size_t capacity) noexcept {
    if (error != nullptr && capacity > 0) {
        error[0] = '\0';
    }
}

template <class Action>
auto guarded(const char* operation, char* error, size_t capacity, Action action,
             decltype(action()) failure) noexcept -> decltype(action()) {
    clear_error(error, capacity);
    try {
        return action();
    } catch (const c10::Error& exception) {
        report(operation, exception.what_without_backtrace(), error, capacity);
    } catch (const std::exception& exception) {
        report(operation, exception.what(), error, capacity);
    } catch (...) {
        report(operation, "unknown C++ exception", error, capacity);
    }
    return failure;
}

// Resolve the CUDA runtime that LibTorch already loaded. CUDA archives can
// ship a hash-suffixed libcudart filename with a different SONAME.
class Library {
public:
    Library() {
#if defined(_WIN32)
        const auto major = CUDART_VERSION / 1000;
        const auto filename = "cudart64_" + std::to_string(major > 11 ? major : major * 10) + ".dll";
        handle_ = GetModuleHandleA(filename.c_str());
#else
        Dl_info info{};
        if (dladdr(reinterpret_cast<const void*>(&at::cuda::graph_pool_handle), &info) != 0) {
            handle_ = dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
        }
#endif
        TORCH_CHECK(handle_ != nullptr, "could not resolve LibTorch's loaded CUDA runtime");
    }

    ~Library() {
#if !defined(_WIN32)
        dlclose(handle_);
#endif
    }

    template <class Function>
    Function resolve(const char* name) const {
#if defined(_WIN32)
        const auto symbol = GetProcAddress(handle_, name);
#else
        const auto symbol = dlsym(handle_, name);
#endif
        TORCH_CHECK(symbol != nullptr, "CUDA runtime is missing ", name);
        return reinterpret_cast<Function>(symbol);
    }

private:
#if defined(_WIN32)
    HMODULE handle_ = nullptr;
#else
    void* handle_ = nullptr;
#endif
};

struct Runtime {
    decltype(&cudaEventCreateWithFlags) event_create;
    decltype(&cudaEventDestroy) event_destroy;
    decltype(&cudaEventRecord) event_record;
    decltype(&cudaGetErrorString) error_string;
    decltype(&cudaGetLastError) last_error;
    decltype(&cudaGraphDestroy) graph_destroy;
    decltype(&cudaStreamEndCapture) capture_end;
    decltype(&cudaStreamIsCapturing) capture_status;
    decltype(&cudaStreamSynchronize) stream_synchronize;
    decltype(&cudaStreamWaitEvent) stream_wait;
};

const Runtime& runtime() {
    static const auto loaded = [] {
        const Library lib;
        return Runtime{
            lib.resolve<decltype(&cudaEventCreateWithFlags)>("cudaEventCreateWithFlags"),
            lib.resolve<decltype(&cudaEventDestroy)>("cudaEventDestroy"),
            lib.resolve<decltype(&cudaEventRecord)>("cudaEventRecord"),
            lib.resolve<decltype(&cudaGetErrorString)>("cudaGetErrorString"),
            lib.resolve<decltype(&cudaGetLastError)>("cudaGetLastError"),
            lib.resolve<decltype(&cudaGraphDestroy)>("cudaGraphDestroy"),
            lib.resolve<decltype(&cudaStreamEndCapture)>("cudaStreamEndCapture"),
            lib.resolve<decltype(&cudaStreamIsCapturing)>("cudaStreamIsCapturing"),
            lib.resolve<decltype(&cudaStreamSynchronize)>("cudaStreamSynchronize"),
            lib.resolve<decltype(&cudaStreamWaitEvent)>("cudaStreamWaitEvent"),
        };
    }();
    return loaded;
}

void warn_cuda(cudaError_t status, const char* operation) noexcept {
    if (status != cudaSuccess) {
        report(operation, runtime().error_string(status), nullptr, 0);
        (void)runtime().last_error();
    }
}

void require_idle(CUDAStream stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    C10_CUDA_CHECK(runtime().capture_status(stream.stream(), &status));
    TORCH_CHECK(status == cudaStreamCaptureStatusNone, "CUDA stream is already capturing");
}

class Pool {
public:
    Pool(c10::DeviceIndex device, c10::cuda::MempoolId_t id) : device_(device), id_(id) {
        allocator::ensureExistsAndIncrefPool(device_, id_);
    }

    ~Pool() {
        guarded("release pool", nullptr, 0, [&] {
            allocator::releasePool(device_, id_);
            return 0;
        }, 1);
    }

    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    c10::cuda::MempoolId_t id() const { return id_; }

    int use_count() const { return allocator::getPoolUseCount(device_, id_); }

private:
    c10::DeviceIndex device_;
    c10::cuda::MempoolId_t id_;
};

class Fence {
public:
    explicit Fence(c10::DeviceIndex device) : device_(device) {
        C10_CUDA_CHECK(runtime().event_create(&event_, cudaEventDisableTiming));
    }

    ~Fence() {
        guarded("delete fence", nullptr, 0, [&] {
            c10::DeviceGuard guard(c10::Device(c10::DeviceType::CUDA, device_));
            warn_cuda(runtime().event_destroy(event_), "delete fence");
            return 0;
        }, 1);
    }

    Fence(const Fence&) = delete;
    Fence& operator=(const Fence&) = delete;

    void order(CUDAStream from, CUDAStream to) const {
        if (from != to) {
            C10_CUDA_CHECK(runtime().event_record(event_, from.stream()));
            C10_CUDA_CHECK(runtime().stream_wait(to.stream(), event_, 0));
        }
    }

private:
    c10::DeviceIndex device_;
    cudaEvent_t event_ = nullptr;
};

// LibTorch 2.7 ends allocator and RNG capture only after a successful
// cudaStreamEndCapture. Undo the capture bookkeeping when a graph is rejected.
class ManagedGraph : public at::cuda::CUDAGraph {
public:
    void abandon(CUDAStream stream, const Pool& pool, int prior_uses) noexcept {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (runtime().capture_status(stream.stream(), &status) == cudaSuccess
            && status != cudaStreamCaptureStatusNone) {
            cudaGraph_t discarded = nullptr;
            const auto ended = runtime().capture_end(stream.stream(), &discarded);
            if (ended != cudaSuccess && ended != cudaErrorStreamCaptureInvalidated) {
                warn_cuda(ended, "cancel capture");
            }
            if (discarded != nullptr) {
                warn_cuda(runtime().graph_destroy(discarded), "delete canceled capture");
            }
        }
        // A capture error must not poison the next eager CUDA operation.
        (void)runtime().last_error();
        for (auto state = captured_generator_states_.begin();
             state != captured_generator_states_.end();) {
            if (state->first->registered_graphs_.count(this) == 0) {
                // An allocation error can interrupt LibTorch during RNG registration.
                state = captured_generator_states_.erase(state);
            } else {
                state->first->capturing_ = false;
                ++state;
            }
        }
        if (!has_graph_ && !has_graph_exec_) {
            if (graph_ != nullptr) {
                warn_cuda(runtime().graph_destroy(graph_), "delete rejected capture");
                graph_ = nullptr;
            }
            guarded("release rejected capture", nullptr, 0, [&] {
                if (pool.use_count() > prior_uses) {
                    // A late error can occur after LibTorch already ended its
                    // allocation scope. Only its captured pool reference remains.
                    try {
                        allocator::endAllocateToPool(stream.device_index(), pool.id());
                    } catch (const c10::Error&) {
                    }
                    allocator::releasePool(stream.device_index(), pool.id());
                }
                return 0;
            }, 1);
        }
    }
};

std::atomic_flag capturing = ATOMIC_FLAG_INIT;
thread_local bool side_stream_entered = false;

class Graph {
public:
    explicit Graph(c10::DeviceIndex device, const Graph* shared)
        : device_(device),
          side_(shared == nullptr ? c10::cuda::getStreamFromPool(false, device) : shared->side_),
          pool_(device, shared == nullptr ? at::cuda::graph_pool_handle() : shared->pool_.id()),
          fence_(device),
          owner_(std::this_thread::get_id()) {}

    void enter() {
        require_owner();
        TORCH_CHECK(state_ == State::fresh, "only a new graph can warm up");
        enter_side();
        state_ = State::warming;
    }

    void leave() {
        require_owner();
        TORCH_CHECK(state_ == State::warming, "CUDA graph warmup is not active");
        state_ = State::fresh;
        leave_side();
    }

    void begin() {
        require_owner();
        TORCH_CHECK(state_ == State::fresh, "a graph can be captured once");
        TORCH_CHECK(!side_stream_entered, "another CUDA graph scope is active on this thread");
        TORCH_CHECK(!capturing.test_and_set(std::memory_order_acquire),
                    "another RVC CUDA graph capture is active");
        try {
            enter_side();
            C10_CUDA_CHECK(runtime().stream_synchronize(side_.stream()));
            prior_pool_uses_ = pool_.use_count();
            try {
                graph_.capture_begin(pool_.id(), cudaStreamCaptureModeThreadLocal);
            } catch (...) {
                graph_.abandon(side_, pool_, prior_pool_uses_);
                throw;
            }
            state_ = State::capturing;
        } catch (...) {
            state_ = State::failed;
            restore();
            capturing.clear(std::memory_order_release);
            throw;
        }
    }

    void end() {
        require_owner();
        TORCH_CHECK(state_ == State::capturing, "no CUDA graph capture is active");
        try {
            graph_.capture_end();
        } catch (...) {
            abandon();
            throw;
        }
        capturing.clear(std::memory_order_release);
        state_ = State::failed;
        leave_side();
        state_ = State::ready;
    }

    void replay() {
        require_owner();
        TORCH_CHECK(state_ == State::ready, "CUDA graph was not captured successfully");
        TORCH_CHECK(!side_stream_entered, "cannot replay a CUDA graph inside a graph scope");
        const auto stream = c10::cuda::getCurrentCUDAStream(device_);
        TORCH_CHECK(!replay_stream_ || stream == *replay_stream_,
                    "the replay stream changed for this graph");
        replay_stream_ = stream;
        // LibTorch launches into the caller's current stream for the graph
        // device. Copies into and reads from static graph tensors use it too.
        graph_.replay();
    }

    void validate_sharing(c10::DeviceIndex device) const {
        require_owner();
        TORCH_CHECK(device == device_ && state_ == State::ready,
                    "share the pool of a captured graph on the same CUDA device");
    }

    void validate_close() const {
        if (side_guard_) {
            require_owner();
        }
    }

    void close() {
        if (state_ == State::capturing) {
            abandon();
        } else if (state_ == State::warming) {
            leave();
        }
        // A graph can still be executing asynchronously at destruction.
        if (replay_stream_) {
            c10::DeviceGuard guard(c10::Device(c10::DeviceType::CUDA, device_));
            C10_CUDA_CHECK(runtime().stream_synchronize(replay_stream_->stream()));
        }
    }

private:
    enum class State { fresh, warming, capturing, ready, failed };

    void require_owner() const {
        TORCH_CHECK(owner_ == std::this_thread::get_id(), "CUDA graph used by another thread");
    }

    void enter_side() {
        TORCH_CHECK(!side_stream_entered, "another CUDA graph scope is active on this thread");
        const auto caller = c10::cuda::getCurrentCUDAStream(device_);
        side_guard_.emplace(side_);
        side_stream_entered = true;
        caller_stream_ = caller;
        try {
            require_idle(side_);
            fence_.order(caller, side_);
        } catch (...) {
            restore();
            throw;
        }
    }

    void leave_side() {
        try {
            fence_.order(side_, *caller_stream_);
        } catch (...) {
            restore();
            throw;
        }
        restore();
    }

    void restore() noexcept {
        if (side_guard_) {
            side_guard_.reset();
            caller_stream_.reset();
            side_stream_entered = false;
        }
    }

    void abandon() noexcept {
        graph_.abandon(side_, pool_, prior_pool_uses_);
        state_ = State::failed;
        restore();
        capturing.clear(std::memory_order_release);
    }

    c10::DeviceIndex device_;
    CUDAStream side_;
    Pool pool_;
    Fence fence_;
    std::thread::id owner_;
    State state_ = State::fresh;
    std::optional<c10::StreamGuard> side_guard_;
    std::optional<CUDAStream> caller_stream_;
    std::optional<CUDAStream> replay_stream_;
    int prior_pool_uses_ = 0;
    // Release LibTorch's pool reference before Pool releases the bridge's own.
    ManagedGraph graph_;
};

Graph& checked(void* handle) {
    TORCH_CHECK(handle != nullptr, "null CUDA graph handle");
    return *static_cast<Graph*>(handle);
}

template <void (Graph::*Method)()>
int32_t call(void* handle, char* error, size_t capacity, const char* operation) noexcept {
    return guarded(operation, error, capacity, [&] {
        (checked(handle).*Method)();
        return 0;
    }, 1);
}

} // namespace

extern "C" void* rvc_cuda_graph_create(
    int32_t device, const void* shared, char* error, size_t capacity) noexcept {
    return guarded("create", error, capacity, [&]() -> void* {
        TORCH_CHECK(!side_stream_entered, "cannot create a CUDA graph inside a graph scope");
        TORCH_CHECK(device >= 0 && device < c10::cuda::device_count(),
                    "CUDA graph device ", device, " is unavailable");
        const auto index = static_cast<c10::DeviceIndex>(device);
        const auto* parent = static_cast<const Graph*>(shared);
        if (parent != nullptr) {
            parent->validate_sharing(index);
        }
        at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
        c10::DeviceGuard guard(c10::Device(c10::DeviceType::CUDA, index));
        return new Graph(index, parent);
    }, nullptr);
}

extern "C" int32_t rvc_cuda_graph_enter(void* handle, char* error, size_t capacity) noexcept {
    return call<&Graph::enter>(handle, error, capacity, "warm up");
}

extern "C" int32_t rvc_cuda_graph_leave(void* handle, char* error, size_t capacity) noexcept {
    return call<&Graph::leave>(handle, error, capacity, "finish warmup");
}

extern "C" int32_t rvc_cuda_graph_begin(void* handle, char* error, size_t capacity) noexcept {
    return call<&Graph::begin>(handle, error, capacity, "begin capture");
}

extern "C" int32_t rvc_cuda_graph_end(void* handle, char* error, size_t capacity) noexcept {
    return call<&Graph::end>(handle, error, capacity, "end capture");
}

extern "C" int32_t rvc_cuda_graph_replay(void* handle, char* error, size_t capacity) noexcept {
    return call<&Graph::replay>(handle, error, capacity, "replay");
}

extern "C" int32_t rvc_cuda_graph_trim(char* error, size_t capacity) noexcept {
    return guarded("trim cache", error, capacity, [&] {
        TORCH_CHECK(!side_stream_entered, "cannot trim CUDA cache inside a graph scope");
        TORCH_CHECK(!capturing.test_and_set(std::memory_order_acquire),
                    "cannot trim CUDA cache during graph capture");
        struct Release {
            ~Release() { capturing.clear(std::memory_order_release); }
        } release;
        allocator::emptyCache();
        return 0;
    }, 1);
}

extern "C" int32_t rvc_cuda_graph_delete(void* handle, char* error, size_t capacity) noexcept {
    return guarded("delete", error, capacity, [&] {
        if (handle != nullptr) {
            auto& graph = checked(handle);
            graph.validate_close();
            std::unique_ptr<Graph> deleting(&graph);
            graph.close();
        }
        return 0;
    }, 1);
}
