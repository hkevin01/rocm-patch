// RDNA1 Conv2d Override - Force Non-Coherent Memory
// This extension intercepts Conv2d operations and uses non-coherent memory allocations
// Compatible with RDNA1 (gfx1010) which lacks fine-grained SVM support

#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Override hipMalloc to use non-coherent memory
void* rdna1_malloc(size_t size) {
    void* ptr = nullptr;
    
    // Try to allocate with non-coherent flag
    // Use hipExtMallocWithFlags if available, otherwise fall back to regular hipMalloc
    hipError_t err = hipExtMallocWithFlags(&ptr, size, hipHostMallocNonCoherent);
    
    if (err != hipSuccess) {
        // Fallback to regular malloc if extension not available
        err = hipMalloc(&ptr, size);
    }
    
    if (err != hipSuccess) {
        throw std::runtime_error("Failed to allocate non-coherent memory for RDNA1");
    }
    
    return ptr;
}

// Custom Conv2d implementation that avoids MIOpen's coherent memory
torch::Tensor rdna1_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    // Guard to ensure we're on the right device
    c10::cuda::CUDAGuard device_guard(input.device());
    
    // For now, we'll use a CPU-based fallback for the actual convolution
    // This avoids MIOpen entirely
    // TODO: Implement custom GPU kernel with non-coherent memory
    
    auto input_cpu = input.to(torch::kCPU);
    auto weight_cpu = weight.to(torch::kCPU);
    torch::optional<torch::Tensor> bias_cpu = bias.has_value() 
        ? torch::optional<torch::Tensor>(bias.value().to(torch::kCPU))
        : torch::nullopt;
    
    // Use PyTorch's CPU convolution (which doesn't use MIOpen)
    auto output_cpu = torch::conv2d(
        input_cpu,
        weight_cpu,
        bias_cpu,
        c10::IntArrayRef(stride),
        c10::IntArrayRef(padding),
        c10::IntArrayRef(dilation),
        groups
    );
    
    // Move result back to GPU
    return output_cpu.to(input.device());
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_forward", &rdna1_conv2d_forward, "RDNA1-compatible Conv2d forward pass");
    m.doc() = "Conv2d implementation that avoids MIOpen's cache-coherent memory allocations";
}
