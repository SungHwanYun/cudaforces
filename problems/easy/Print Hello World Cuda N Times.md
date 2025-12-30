# Print Hello World Cuda N Times

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Print the following line exactly **n** times, each on a separate line:
```
Hello World Cuda
```

### Input
A single integer **n** (1 ≤ n ≤ 100).

### Output
Print the following line exactly n times, each on a separate line:
```
Hello World Cuda
```

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 1 | Hello World Cuda |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 10 | Hello World Cuda<br>Hello World Cuda<br>Hello World Cuda<br>Hello World Cuda<br>Hello World Cuda<br>Hello World Cuda<br>Hello World Cuda<br>Hello World Cuda<br>Hello World Cuda<br>Hello World Cuda |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void helloKernel(int* dummy, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        printf("Hello World Cuda\n");
    }
}

int main() {
    int n;
    scanf("%d", &n);
    
    int* d_dummy;
    cudaMalloc(&d_dummy, sizeof(int));
    
    // Launch kernel with n threads (max 100)
    helloKernel<<<1, n>>>(d_dummy, n);
    
    cudaDeviceSynchronize();
    cudaFree(d_dummy);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void helloKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaFree` |
| ✅ Kernel called | `<<<1, n>>>` launches n threads |
| ✅ Meaningful computation | Threads print based on input |

---

## CUDA Concepts Covered

### 1. Reading Input from Host

Input is read on the **host (CPU)** side using standard `scanf`:

```cuda
int main() {
    int n;
    scanf("%d", &n);  // Host reads input
    
    // Pass n to kernel
    helloKernel<<<1, n>>>(d_dummy, n);
}
```

### 2. Passing Parameters to Kernels

Kernels can receive parameters just like regular functions:

```cuda
__global__ void helloKernel(int* dummy, int n) {
    //                                   ↑
    //                         Parameter passed from host
    if (threadIdx.x < n) {
        // Use n for bounds checking
    }
}
```

### 3. Dynamic Thread Count

The number of threads can be determined at runtime:

```cuda
helloKernel<<<1, n>>>(d_dummy, n);
//              ↑
//    Variable number of threads based on input
```

This allows the kernel to adapt to different input sizes.

### 4. Thread-to-Work Mapping

```
Input n = 5:

Threads:  [T0] [T1] [T2] [T3] [T4]
           │    │    │    │    │
Output:   "Hello World Cuda" × 5
```

Each thread is responsible for one output line.

### 5. Host-Device Communication Flow

```
┌────────────────────────────────────────────────────────┐
│                      HOST (CPU)                        │
│  1. scanf("%d", &n)     ← Read input                   │
│  2. cudaMalloc(...)     ← Allocate GPU memory          │
│  3. kernel<<<1,n>>>     ← Launch kernel with n threads │
│  4. cudaDeviceSynchronize() ← Wait for completion      │
│  5. cudaFree(...)       ← Clean up                     │
└────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│                    DEVICE (GPU)                        │
│  Each thread (0 to n-1):                               │
│    if (threadIdx.x < n)                                │
│        printf("Hello World Cuda\n")                    │
└────────────────────────────────────────────────────────┘
```

---

## Common Mistakes

### ❌ Not Passing n to Kernel
```cuda
__global__ void helloKernel(int* dummy) {
    // How do we know the bound without n?
    printf("Hello World Cuda\n");  // All threads print!
}
```

### ❌ Reading Input Inside Kernel
```cuda
__global__ void helloKernel(int* dummy) {
    int n;
    scanf("%d", &n);  // ERROR: scanf doesn't work in device code!
}
```

### ❌ Hardcoding Thread Count
```cuda
helloKernel<<<1, 100>>>(d_dummy, n);  // Always 100 threads regardless of n
```
This wastes resources and requires careful bounds checking.

### ❌ Exceeding Maximum Threads Per Block
```cuda
// If n > 1024, this will fail!
helloKernel<<<1, n>>>(d_dummy, n);
```
For large n, use multiple blocks: `<<<(n+255)/256, 256>>>`

---

## Key Takeaways

1. **Input is read on the host** using standard C I/O functions
2. **Parameters can be passed** from host to kernel
3. **Thread count can be dynamic** based on runtime values
4. **Bounds checking** ensures only required threads execute
5. **Host-device flow**: Read → Allocate → Launch → Sync → Free

---

## Practice Exercises

1. What happens if you input n = 0? Add error handling.
2. Modify to handle n > 1024 using multiple blocks
3. Print with thread number: `printf("Line %d: Hello World Cuda\n", threadIdx.x);`

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/3)*
