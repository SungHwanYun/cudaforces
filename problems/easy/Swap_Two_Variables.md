# Swap Two Variables

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given two integers. Swap their values and output the result.

### Input
A single line containing two integers, a₁ and a₂, separated by a space.

### Output
Print the values of **a₁** and **a₂** after swapping them, separated by a space.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 1 2 | 2 1 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 3 7 | 7 3 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 10 25 | 25 10 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void swapKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int temp = data[0];
        data[0] = data[1];
        data[1] = temp;
    }
}

int main() {
    int a1, a2;
    scanf("%d %d", &a1, &a2);
    
    // Host array
    int h_data[2] = {a1, a2};
    
    // Device memory
    int* d_data;
    cudaMalloc(&d_data, 2 * sizeof(int));
    
    // Copy to device
    cudaMemcpy(d_data, h_data, 2 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel with 1 thread
    swapKernel<<<1, 1>>>(d_data);
    
    // Copy back to host
    cudaMemcpy(h_data, d_data, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d %d\n", h_data[0], h_data[1]);
    
    cudaFree(d_data);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void swapKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs swap operation on GPU memory |

---

## CUDA Concepts Covered

### 1. Passing Data via Device Memory

Unlike simple scalar parameters, we pass data through GPU memory:

```cuda
int h_data[2] = {a1, a2};      // Host array
int* d_data;
cudaMalloc(&d_data, 2 * sizeof(int));  // Allocate on device
cudaMemcpy(d_data, h_data, 2 * sizeof(int), cudaMemcpyHostToDevice);  // Copy to device
```

### 2. Single Thread Operation

Some operations don't need parallelism but still must follow CUDA patterns:

```cuda
__global__ void swapKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {  // Only thread 0 performs the swap
        int temp = data[0];
        data[0] = data[1];
        data[1] = temp;
    }
}
```

### 3. Memory Transfer Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        HOST (CPU)                               │
│  1. scanf("%d %d", &a1, &a2)   ← Read input                     │
│  2. h_data[2] = {a1, a2}       ← Store in host array            │
│  3. cudaMalloc(&d_data, ...)   ← Allocate GPU memory            │
│  4. cudaMemcpy(...HostToDevice)← Copy data to GPU               │
│  5. swapKernel<<<1, 1>>>       ← Launch kernel                  │
│  6. cudaMemcpy(...DeviceToHost)← Copy result back               │
│  7. printf(...)                ← Output result                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DEVICE (GPU)                              │
│  Thread 0:                                                      │
│    temp = data[0]                                               │
│    data[0] = data[1]                                            │
│    data[1] = temp                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Round-Trip Data Transfer

This problem demonstrates the complete data lifecycle:

```
Host Data → Device Memory → Kernel Computation → Device Memory → Host Data
   [1,2]  →     [1,2]     →      swap()        →     [2,1]     →   [2,1]
```

### 5. Using Arrays for Multiple Values

When passing multiple related values, arrays are often cleaner:

```cuda
// Instead of separate variables:
int *d_a1, *d_a2;  // Messy

// Use an array:
int* d_data;  // data[0] = a1, data[1] = a2
cudaMalloc(&d_data, 2 * sizeof(int));
```

---

## Common Mistakes

### ❌ Swapping on Host Only
```cuda
int main() {
    int a1, a2;
    scanf("%d %d", &a1, &a2);
    
    // This swaps on CPU, not GPU!
    int temp = a1;
    a1 = a2;
    a2 = temp;
    
    printf("%d %d\n", a1, a2);  // No CUDA used!
}
```

### ❌ Forgetting to Copy Back
```cuda
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
swapKernel<<<1, 1>>>(d_data);
// Missing: cudaMemcpy back to host!
printf("%d %d\n", h_data[0], h_data[1]);  // Still original values!
```

### ❌ Not Using GPU Memory
```cuda
__global__ void swapKernel(int* a, int* b) {
    // Error: a and b point to host memory!
}

int main() {
    int a1 = 1, a2 = 2;
    swapKernel<<<1, 1>>>(&a1, &a2);  // Cannot pass host pointers!
}
```

### ❌ All Threads Performing Swap
```cuda
__global__ void swapKernel(int* data) {
    // Without idx check, all threads swap (redundant work)
    int temp = data[0];
    data[0] = data[1];
    data[1] = temp;
}
```

---

## Key Takeaways

1. **Host ↔ Device transfer** is required for data to be processed on GPU
2. **Even simple operations** must follow CUDA memory patterns
3. **Arrays simplify** passing multiple related values
4. **Thread guards** (`if (idx == 0)`) prevent redundant operations
5. **Complete flow**: Allocate → Copy to device → Execute → Copy back → Free

---

## Practice Exercises

1. Modify to swap 3 variables in a circular manner (a→b→c→a)
2. Implement parallel swap for an array of pairs
3. Add error checking for `cudaMalloc` and `cudaMemcpy` return values

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/7)*
