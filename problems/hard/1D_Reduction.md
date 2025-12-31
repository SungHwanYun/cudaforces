# 1D Reduction

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Medium | 128 MB | 1 s | MenOfPassion |

## Problem Description

Reduction is a fundamental parallel computing pattern that combines all elements of an array into a single value using an associative operator. Given an array **A** of **n** integers, compute the sum of all elements using parallel reduction.

### Input
The first line contains an integer n (1 ≤ n ≤ 100), the size of array A.

The second line contains n space-separated integers aᵢ (-1000 ≤ aᵢ ≤ 1000, 0 ≤ i ≤ n-1) representing the elements of array A.

### Output
Print a single integer: the sum of all elements in array A.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 5<br>1 2 3 4 5 | 15 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 4<br>-10 20 -30 40 | 20 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 6<br>1000 -1000 1000 -1000 500 -500 | 0 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void reductionKernel(int* A, int* result, int n) {
    __shared__ int sdata[128];  // Shared memory for reduction
    
    int tid = threadIdx.x;
    
    // Load data into shared memory
    if (tid < n) {
        sdata[tid] = A[tid];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < n) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        *result = sdata[0];
    }
}

int main() {
    int n;
    scanf("%d", &n);
    
    int* h_A = (int*)malloc(n * sizeof(int));
    
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_A[i]);
    }
    
    // Device memory
    int *d_A, *d_result;
    cudaMalloc(&d_A, n * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    // Copy input to device
    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 128;
    reductionKernel<<<1, threadsPerBlock>>>(d_A, d_result, n);
    
    // Copy result back to host
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d\n", result);
    
    free(h_A);
    cudaFree(d_A);
    cudaFree(d_result);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void reductionKernel()` |
| ✅ Uses parallelism | Parallel reduction with stride pattern |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Uses shared memory | `__shared__ int sdata[]` for fast access |
| ✅ Kernel called | `<<<1, 128>>>` launches threads |
| ✅ Meaningful computation | Performs parallel sum reduction on GPU |

---

## CUDA Concepts Covered

### 1. What is Reduction?

Reduction combines n values into 1 value using an operator (sum, max, min, etc.):

```
[1, 2, 3, 4, 5] → sum → 15
```

### 2. Parallel Reduction Algorithm

Instead of sequential O(n), use tree-based parallel reduction O(log n):

```
Initial:  [1]  [2]  [3]  [4]  [5]  [0]  [0]  [0]

Step 1 (stride=4):
          [1]  [2]  [3]  [4]  [5]  [0]  [0]  [0]
           ↓         ↓
          +5        +0
           ↓         ↓
          [6]  [2]  [3]  [4]  [5]  [0]  [0]  [0]

Step 2 (stride=2):
          [6]  [2]  [3]  [4]
           ↓    ↓
          +3   +4
           ↓    ↓
          [9]  [6]  [3]  [4]

Step 3 (stride=1):
          [9]  [6]
           ↓
          +6
           ↓
          [15]

Result: 15
```

### 3. Shared Memory

Shared memory is fast on-chip memory accessible by all threads in a block:

```cuda
__shared__ int sdata[128];  // Declare shared memory

// Each thread loads one element
sdata[tid] = A[tid];

__syncthreads();  // Wait for all threads to finish loading
```

### 4. Thread Synchronization

`__syncthreads()` ensures all threads reach this point before continuing:

```cuda
// All threads must complete loading before reduction starts
__syncthreads();

// All threads must complete each step before next step
for (int stride = ...; stride > 0; stride >>= 1) {
    // ... reduction step ...
    __syncthreads();
}
```

### 5. Stride Pattern

Each step halves the stride, combining pairs of elements:

```cuda
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
}
```

---

## Visualization (Example 1)

```
Input: [1, 2, 3, 4, 5] (n=5)
Padded to power of 2: [1, 2, 3, 4, 5, 0, 0, 0]

Step 1: stride = 4
Thread 0: sdata[0] += sdata[4] → 1 + 5 = 6
Thread 1: sdata[1] += sdata[5] → 2 + 0 = 2
Thread 2: sdata[2] += sdata[6] → 3 + 0 = 3
Thread 3: sdata[3] += sdata[7] → 4 + 0 = 4
Result: [6, 2, 3, 4, ...]

Step 2: stride = 2
Thread 0: sdata[0] += sdata[2] → 6 + 3 = 9
Thread 1: sdata[1] += sdata[3] → 2 + 4 = 6
Result: [9, 6, ...]

Step 3: stride = 1
Thread 0: sdata[0] += sdata[1] → 9 + 6 = 15
Result: [15, ...]

Final: sdata[0] = 15 ✓
```

---

## Alternative Solutions

### Simple Sequential (Single Thread)

```cuda
__global__ void reductionKernel(int* A, int* result, int n) {
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += A[i];
        }
        *result = sum;
    }
}
```

### Using atomicAdd (Not Recommended for Reduction)

```cuda
__global__ void reductionKernel(int* A, int* result, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        atomicAdd(result, A[tid]);  // Atomic but serialized
    }
}
```

### Simpler Parallel Reduction

```cuda
__global__ void reductionKernel(int* A, int* result, int n) {
    __shared__ int sdata[128];
    int tid = threadIdx.x;
    
    sdata[tid] = (tid < n) ? A[tid] : 0;
    __syncthreads();
    
    // Simple loop reduction
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) *result = sdata[0];
}
```

---

## Sequential vs Parallel Reduction

| Approach | Time Complexity | Steps for n=8 |
|----------|-----------------|---------------|
| Sequential | O(n) | 7 additions |
| **Parallel** | O(log n) | 3 steps |

```
Sequential:  1+2+3+4+5+6+7+8 = 36  (7 additions, serial)

Parallel:    Step 1: 4 additions in parallel
             Step 2: 2 additions in parallel
             Step 3: 1 addition
             Total: 3 steps with parallelism
```

---

## Shared Memory Benefits

| Memory Type | Latency | Location | Scope |
|-------------|---------|----------|-------|
| Global | ~400 cycles | DRAM | All threads |
| **Shared** | ~5 cycles | On-chip | Block only |
| Registers | ~1 cycle | On-chip | Thread only |

Shared memory is ~80x faster than global memory!

---

## Common Mistakes

### ❌ Missing syncthreads
```cuda
// DANGEROUS - Race condition!
for (int stride = ...; stride > 0; stride >>= 1) {
    sdata[tid] += sdata[tid + stride];
    // Missing __syncthreads()!
}
```

### ❌ Out-of-Bounds Access
```cuda
// Wrong - doesn't check bounds
sdata[tid] += sdata[tid + stride];

// Correct - check bounds
if (tid < stride && tid + stride < n) {
    sdata[tid] += sdata[tid + stride];
}
```

### ❌ Not Initializing Unused Threads
```cuda
// Wrong - garbage values
sdata[tid] = A[tid];

// Correct - initialize to 0 for unused
sdata[tid] = (tid < n) ? A[tid] : 0;
```

---

## Reduction Operations

The same pattern works for other associative operations:

| Operation | Identity | Usage |
|-----------|----------|-------|
| **Sum** | 0 | Total, average |
| Product | 1 | Factorial |
| Max | -∞ | Finding maximum |
| Min | +∞ | Finding minimum |
| AND | true | All true? |
| OR | false | Any true? |

```cuda
// Max reduction
if (tid < stride) {
    sdata[tid] = max(sdata[tid], sdata[tid + stride]);
}
```

---

## Key Takeaways

1. **Reduction** — combines n values into 1 using associative operator
2. **Parallel reduction** — O(log n) instead of O(n)
3. **Shared memory** — fast on-chip storage for thread block
4. **__syncthreads()** — essential for correct parallel execution
5. **Stride pattern** — halve stride each step

---

## Practice Exercises

1. Implement **max reduction** — find maximum element
2. Implement **min reduction** — find minimum element
3. Implement **product reduction** — multiply all elements
4. Extend to **multiple blocks** for larger arrays

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/49)*
