# Print Hello World Cuda Ten Times

| Difficulty | Memory Limit | Time Limit |
|------------|--------------|------------|
| Easy | 128 MB | 1 s |

## Problem Description

Print the phrase **"Hello World Cuda"** exactly ten times. Each occurrence should be printed on its own line.

### Input

There is no input.

### Output

Output the line below exactly ten times, each on a separate line:

Hello World Cuda

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
|  | Hello World Cuda<br>Hello World Cuda<br>Hello World Cuda<br>Hello World Cuda<br>Hello World Cuda<br>Hello World Cuda<br>Hello World Cuda<br>Hello World Cuda<br>Hello World Cuda<br>Hello World Cuda |

---

## Solution Code

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](https://github.com/SungHwanYun/cudaforces/blob/main/GUIDE.md) first.

```cuda
__global__ void solve(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // TODO: Implement solution
    }
}

int main() {
    int n;
    scanf("%d", &n);
    
    int* h_data = (int*)malloc(n * sizeof(int));
    // TODO: Read input
    
    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    solve<<<numBlocks, blockSize>>>(d_data, n);
    
    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // TODO: Print output
    
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}
```

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__` function defined |
| ✅ Uses parallelism | `threadIdx.x` and `blockIdx.x` used |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<blocks, threads>>>` syntax |
| ✅ Meaningful computation | Problem-specific logic implemented |

---

## CUDA Concepts Covered

### Basic Concepts

This problem covers fundamental CUDA concepts:

- **Kernel Declaration**: Using `__global__` to define GPU functions
- **Thread Indexing**: Using `threadIdx.x` and `blockIdx.x` to identify threads
- **Memory Management**: `cudaMalloc`, `cudaMemcpy`, `cudaFree`
- **Kernel Launch**: `<<<blocks, threads>>>` syntax

---

## Key Takeaways

1. Understand the problem requirements before coding
2. Map the problem to parallel threads appropriately
3. Always perform bounds checking in kernels
4. Follow the host-device memory transfer pattern
5. Test with the provided examples

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/2)*
