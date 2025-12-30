# CUDA Online Judge Guide

Welcome to **CUDA Online Judge (cudaforces.com)** - an educational platform for learning GPU programming!

---

## About CUDA Online Judge

CUDA Online Judge is designed for **educational purposes**. To help users learn proper GPU programming patterns:

### Minimal Library Policy

Only essential libraries are allowed to encourage learning C-style memory management and algorithm implementation from scratch. This means:
- No `std::vector`, `std::string`, or other STL containers
- No `qsort`, `strcpy`, or similar convenience functions
- You'll implement algorithms yourself using basic C constructs

### Parallelism Validation

The system performs validation checks to ensure your code follows proper CUDA parallel programming patterns:

| Validation Item | Description | Error Code |
|-----------------|-------------|------------|
| ✅ Kernel exists | At least one `__global__` function required | E3001 |
| ✅ Meaningful computation | Kernel must perform actual computation | E3002 |
| ✅ Uses parallelism | Must use `threadIdx`, `blockIdx`, etc. | E3003 |
| ✅ Uses GPU memory | Must use `cudaMalloc`, `cudaMemcpy`, etc. | E3004 |
| ✅ Kernel called | Must call kernel with `<<<>>>` syntax | E3001 |
| ❌ No forbidden functions | Cannot use `qsort`, `strcpy`, etc. | E3005 |
| ❌ No forbidden types | Cannot use `std::vector`, `std::string`, etc. | E3006 |

These requirements ensure that users learn authentic CUDA programming skills rather than bypassing GPU concepts.

### CPU Transpiler System

CUDA Online Judge uses a **CPU Transpiler** that converts your CUDA code to run on CPU without requiring actual GPU hardware:

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  CUDA Code  │ ---> │ Transpiler  │ ---> │  C++ Code   │ ---> │ CPU Execute │
│  (Submit)   │      │             │      │  (OpenMP)   │      │  & Judge    │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
```

> ⚠️ **Important**: This system is for **correctness verification** only. Performance benchmarking is not available since CPU emulation behaves differently from actual GPU execution.

---

## Automatically Included Libraries

CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. **You don't need to write any `#include` in your code.**

| Library | Description |
|---------|-------------|
| `stdio.h` / `cstdio` | Standard I/O (`printf`, `scanf`) |
| `stdlib.h` / `cstdlib` | Memory allocation (`malloc`, `free`) |
| `string.h` / `cstring` | Limited functions (`memset`, `memcpy`) |
| `math.h` / `cmath` | Math functions (`sqrt`, `sin`, `cos`) |
| `stdint.h` / `cstdint` | Fixed-width integers (`int32_t`, `uint64_t`) |
| `stdbool.h` / `cstdbool` | Boolean type |
| `float.h` / `cfloat` | Float limits |
| `limits.h` / `climits` | Integer limits |
| `cuda_runtime.h` | CUDA runtime API |

---

## Judgment Results

| Result | Description |
|--------|-------------|
| ✅ **Accepted (AC)** | Correct - All test cases passed |
| ❌ **Wrong Answer (WA)** | Incorrect - Output differs from expected |
| ⚠️ **Compile Error (CE)** | Compilation error - Syntax error |
| ⏱️ **Time Limit Exceeded (TLE)** | Time limit exceeded |
| 💾 **Memory Limit Exceeded (MLE)** | Memory limit exceeded |
| 🚫 **Runtime Error (RE)** | Runtime error - Segfault, etc. |
| 🔒 **Validation Error (VE)** | Validation failed - CUDA rule violation |

---

## Basic Code Template

Here's a minimal template that passes all validation checks:

```cuda
__global__ void kernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Your parallel computation here
    }
}

int main() {
    int n;
    scanf("%d", &n);
    
    // Host memory
    int* h_data = (int*)malloc(n * sizeof(int));
    
    // Device memory
    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    
    // Copy to device
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    kernel<<<numBlocks, blockSize>>>(d_data, n);
    
    // Copy back to host
    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Output results
    // ...
    
    // Free memory
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}
```

---

## Quick Reference

### Kernel Launch Syntax
```cuda
kernel<<<numBlocks, threadsPerBlock>>>(args...);
kernel<<<numBlocks, threadsPerBlock, sharedMemBytes>>>(args...);
```

### Global Thread Index
```cuda
// 1D grid
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D grid
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

### Memory Operations
```cuda
cudaMalloc(&d_ptr, size);                              // Allocate device memory
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);    // Host → Device
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);    // Device → Host
cudaMemset(d_ptr, value, size);                        // Set device memory
cudaFree(d_ptr);                                       // Free device memory
```

### Synchronization
```cuda
__syncthreads();           // Synchronize threads within a block
cudaDeviceSynchronize();   // Synchronize host with device
```

---

## Resources

- 🌐 [CUDA Online Judge](https://cudaforces.com)
- 📧 Contact: ejpark29@gmail.com

---

*Happy CUDA Learning! 🚀*
