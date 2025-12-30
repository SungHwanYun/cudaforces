# Sum of Two Integers

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given two integers a and b. Print the sum a + b.

### Input
The first line contains an integer a.

The second line contains an integer b.

**Constraints:**
- 1 ≤ a, b ≤ 100

### Output
Print the sum a + b on the first line.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 13<br>7 | 20 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 42<br>58 | 100 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 1<br>1 | 2 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void sumKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = *a + *b;
    }
}

int main() {
    int a, b;
    scanf("%d", &a);
    scanf("%d", &b);
    
    // Device memory
    int *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    // Copy inputs to device
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    sumKernel<<<1, 1>>>(d_a, d_b, d_result);
    
    // Copy result back to host
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d\n", result);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void sumKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs addition on GPU |

---

## CUDA Concepts Covered

### 1. GPU Computation Pattern

This is the first problem where we **compute** on the GPU, not just echo:

```cuda
__global__ void sumKernel(int* a, int* b, int* result) {
    *result = *a + *b;  // Actual computation on GPU!
}
```

### 2. Input-Output Memory Pattern

```
Inputs (Host → Device):
  - a: input value 1
  - b: input value 2

Output (Device → Host):
  - result: computed sum
```

### 3. Complete Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  1. scanf a, b           ← Read inputs                   │
│  2. cudaMalloc × 3       ← Allocate d_a, d_b, d_result   │
│  3. cudaMemcpy × 2       ← Copy a, b to device           │
│  4. sumKernel<<<1,1>>>   ← Launch kernel                 │
│  5. cudaMemcpy           ← Copy result back              │
│  6. printf               ← Output result                 │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   d_a: [13]    d_b: [7]                                  │
│          \      /                                        │
│           \    /                                         │
│            +  ← Addition performed here                  │
│            │                                             │
│            ▼                                             │
│   d_result: [20]                                         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 4. Input vs Output Memory Transfers

```cuda
// INPUT: Host → Device (before kernel)
cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

// KERNEL: Computation on device
sumKernel<<<1, 1>>>(d_a, d_b, d_result);

// OUTPUT: Device → Host (after kernel)
cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
```

### 5. Pointer Arithmetic in Kernel

```cuda
*result = *a + *b;
//  ↑      ↑    ↑
// write  read read
// to     from from
// result  a    b
```

---

## Alternative Solutions

### Using Array Instead of Separate Pointers

```cuda
__global__ void sumKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        data[2] = data[0] + data[1];  // result at index 2
    }
}

int main() {
    int a, b;
    scanf("%d", &a);
    scanf("%d", &b);
    
    int h_data[3] = {a, b, 0};  // [a, b, result placeholder]
    
    int* d_data;
    cudaMalloc(&d_data, 3 * sizeof(int));
    cudaMemcpy(d_data, h_data, 3 * sizeof(int), cudaMemcpyHostToDevice);
    
    sumKernel<<<1, 1>>>(d_data);
    
    cudaMemcpy(h_data, d_data, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d\n", h_data[2]);
    
    cudaFree(d_data);
    
    return 0;
}
```

### Computing in Kernel and Printing Directly

```cuda
__global__ void sumKernel(int* a, int* b) {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("%d\n", *a + *b);  // Compute and print in kernel
    }
}

int main() {
    int a, b;
    scanf("%d", &a);
    scanf("%d", &b);
    
    int *d_a, *d_b;
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
    
    sumKernel<<<1, 1>>>(d_a, d_b);
    
    cudaDeviceSynchronize();
    cudaFree(d_a);
    cudaFree(d_b);
    
    return 0;
}
```

---

## Echo vs Compute Comparison

| Aspect | Echo Problems | Compute Problems |
|--------|---------------|------------------|
| Kernel action | Read and print | Read, compute, write |
| Result storage | Not needed | Need output memory |
| Copy back | Optional | Required for result |
| Example | `printf(*a)` | `*result = *a + *b` |

---

## Common Mistakes

### ❌ Forgetting to Copy Result Back
```cuda
sumKernel<<<1, 1>>>(d_a, d_b, d_result);
// Missing cudaMemcpy DeviceToHost!
printf("%d\n", result);  // Undefined value!
```

### ❌ Copying Result Before Kernel Completes
```cuda
sumKernel<<<1, 1>>>(d_a, d_b, d_result);
cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
// This is actually OK - cudaMemcpy synchronizes automatically
// But explicit cudaDeviceSynchronize() is clearer
```

### ❌ Computing on Host Memory
```cuda
__global__ void sumKernel(int* a, int* b, int* result) {
    // Can't access host memory here!
}

int main() {
    int a = 5, b = 7, result;
    sumKernel<<<1, 1>>>(&a, &b, &result);  // Error! Host pointers
}
```

### ❌ Wrong Memory Direction
```cuda
// Want to copy result FROM device TO host
cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyHostToDevice);  // Wrong!
cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);  // Correct!
```

---

## Arithmetic Operations in CUDA

All standard C arithmetic works in CUDA kernels:

| Operation | Syntax | Example |
|-----------|--------|---------|
| Addition | `a + b` | `*c = *a + *b` |
| Subtraction | `a - b` | `*c = *a - *b` |
| Multiplication | `a * b` | `*c = *a * *b` |
| Division | `a / b` | `*c = *a / *b` |
| Modulo | `a % b` | `*c = *a % *b` |

---

## Key Takeaways

1. **First computation problem** — GPU actually does work
2. **Output memory** required to store computed result
3. **Copy result back** — Device → Host after kernel
4. **Three-way data flow** — inputs in, compute, result out
5. **Foundation for all computations** — this pattern scales

---

## Practice Exercises

1. Compute the **difference** a - b
2. Compute the **product** a × b
3. Compute both **sum and product** in one kernel
4. Sum **three** integers

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/21)*
