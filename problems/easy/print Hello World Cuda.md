# Print Hello World Cuda

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

This is a simple output-only problem.

Your task is to write a CUDA program that prints the exact phrase **"Hello World Cuda"**.

### Input
There is no input.

### Output
Print the following text exactly:
```
Hello World Cuda
```

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void helloKernel(int* dummy) {
    if (threadIdx.x == 0) {
        printf("Hello World Cuda\n");
    }
}

int main() {
    // Allocate GPU memory (required for validation)
    int* d_dummy;
    cudaMalloc(&d_dummy, sizeof(int));
    
    // Launch kernel
    helloKernel<<<1, 1>>>(d_dummy);
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Free GPU memory
    cudaFree(d_dummy);
    
    return 0;
}
```

### About CUDA Online Judge

CUDA Online Judge is designed for **educational purposes**. To help users learn proper GPU programming patterns:

1. **Minimal Library Policy**: Only essential libraries are allowed to encourage learning C-style memory management and algorithm implementation from scratch.

2. **Parallelism Validation**: The system performs validation checks to ensure your code follows proper CUDA parallel programming patterns:
   - Must have at least one `__global__` kernel function
   - Must use parallelism variables (`threadIdx`, `blockIdx`, etc.)
   - Must use GPU memory management (`cudaMalloc`, `cudaMemcpy`, etc.)

These requirements ensure that users learn authentic CUDA programming skills rather than bypassing GPU concepts.

### Automatically Included Libraries

CUDA Online Judge automatically includes the following libraries:

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
| `cuda_runtime.h` | CUDA runtime API (auto-included) |

### Why This Code Structure?

CUDA Online Judge requires code to pass validation checks:

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void helloKernel()` |
| ✅ Uses parallelism | `threadIdx.x` in condition |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launch syntax |
| ✅ Meaningful computation | `printf()` output |

---

## CUDA Concepts Covered

### 1. Basic CUDA Program Structure

A CUDA program consists of two main parts:
- **Host code**: Runs on the CPU
- **Device code**: Runs on the GPU

```
┌─────────────────────────────────────────┐
│              CUDA Program               │
├─────────────────┬───────────────────────┤
│   Host (CPU)    │    Device (GPU)       │
│                 │                       │
│  - main()       │  - __global__ kernels │
│  - Memory mgmt  │  - __device__ funcs   │
│  - Kernel launch│  - Parallel execution │
└─────────────────┴───────────────────────┘
```

### 2. The `__global__` Keyword

The `__global__` keyword declares a **kernel function** that:
- Is called from the **host** (CPU)
- Executes on the **device** (GPU)
- Must return `void`

```cuda
__global__ void kernelName() {
    // This code runs on GPU
}
```

### 3. Kernel Launch Syntax: `<<<blocks, threads>>>`

```cuda
helloKernel<<<1, 1>>>(d_dummy);
//           │  │
//           │  └── Threads per block
//           └───── Number of blocks
```

In this problem, we launch with `<<<1, 1>>>`:
- **1 block** containing **1 thread**
- Only one thread executes the kernel

### 4. Built-in Variables for Parallelism

CUDA provides built-in variables to identify each thread:

| Variable | Description |
|----------|-------------|
| `threadIdx.x/y/z` | Thread index within block |
| `blockIdx.x/y/z` | Block index within grid |
| `blockDim.x/y/z` | Threads per block |
| `gridDim.x/y/z` | Blocks in grid |

```cuda
// Only thread 0 executes the print
if (threadIdx.x == 0) {
    printf("Hello World Cuda\n");
}
```

### 5. GPU Memory Management

CUDA requires explicit memory management:

| Function | Description |
|----------|-------------|
| `cudaMalloc` | Allocate GPU memory |
| `cudaMemcpy` | Copy data between host and device |
| `cudaFree` | Free GPU memory |

```cuda
int* d_ptr;
cudaMalloc(&d_ptr, sizeof(int));  // Allocate on GPU
cudaFree(d_ptr);                   // Free GPU memory
```

### 6. `cudaDeviceSynchronize()`

This function blocks the host (CPU) until all previously issued CUDA operations complete.

**Why is it needed?**
- Kernel launches are **asynchronous**
- Without synchronization, `main()` might exit before the kernel finishes
- `printf` from device needs synchronization to flush output

```
Timeline without cudaDeviceSynchronize():
──────────────────────────────────────────►
CPU: main() starts → launch kernel → main() exits
GPU:                  ↓ kernel starts... (may not complete!)

Timeline with cudaDeviceSynchronize():
──────────────────────────────────────────►
CPU: main() starts → launch kernel → wait... → main() exits
GPU:                  ↓ kernel executes → done ↑
```

### 7. Device-side `printf`

CUDA supports `printf` inside device code (compute capability 2.0+).

Key points:
- `stdio.h` is automatically included by CUDA OJ
- Output is buffered and flushed on synchronization
- Useful for debugging but can impact performance

---

## Common Mistakes

### ❌ Missing Synchronization
```cuda
int main() {
    helloKernel<<<1, 1>>>(d_dummy);
    return 0;  // Program may exit before output appears!
}
```

### ❌ Missing GPU Memory Usage
```cuda
// Validation Error E3004: No GPU memory used
int main() {
    helloKernel<<<1, 1>>>();  // No cudaMalloc!
    cudaDeviceSynchronize();
    return 0;
}
```

### ❌ Missing Parallelism Variable
```cuda
// Validation Error E3003: No parallelism used
__global__ void helloKernel() {
    printf("Hello World Cuda\n");  // No threadIdx/blockIdx!
}
```

### ❌ Wrong Output Text
```cuda
printf("Hello World CUDA\n");  // "CUDA" vs "Cuda" - case matters!
```

---

## Key Takeaways

1. **CUDA OJ requires proper CUDA patterns** - even simple problems need kernel, parallelism, and GPU memory
2. **`__global__` functions** are the bridge between CPU and GPU
3. **Kernel launch syntax** `<<<blocks, threads>>>` controls parallelism
4. **Always synchronize** when device output is expected
5. **Exact output matching** is crucial in competitive programming

---

## Practice Exercises

1. Modify the kernel to launch with `<<<1, 5>>>` and observe what happens without the `if` condition
2. Try launching with `<<<5, 1>>>` and use `blockIdx.x` instead of `threadIdx.x`
3. Remove `cudaDeviceSynchronize()` and see if the output still appears

---

## Related Problems
- Problem 002: Hello from Thread (introduces `threadIdx`)
- Problem 003: Hello from Block (introduces `blockIdx`)

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/1)*
