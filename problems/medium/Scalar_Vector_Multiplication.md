# Scalar Vector Multiplication

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Medium | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given an integer vector A of size n and a scalar value k. Compute the scalar multiplication of the vector and output the resulting vector B, where:

$$B[i] = k \times A[i]$$

for all 0 ≤ i < n.

### Input
The first line contains two integers n (1 ≤ n ≤ 100) and k (1 ≤ k ≤ 10,000), the size of the vector and the scalar value.

The second line contains n space-separated integers representing vector A (1 ≤ A[i] ≤ 10,000).

### Output
Output n space-separated integers representing the resulting vector B.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 3 2<br>1 2 3 | 2 4 6 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 5 10<br>5 10 15 20 25 | 50 100 150 200 250 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 1 10000<br>10000 | 100000000 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void scalarMultiplyKernel(int* A, int* B, int k, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        B[idx] = k * A[idx];
    }
}

int main() {
    int n, k;
    scanf("%d %d", &n, &k);
    
    int* h_A = (int*)malloc(n * sizeof(int));
    int* h_B = (int*)malloc(n * sizeof(int));
    
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_A[i]);
    }
    
    // Device memory
    int *d_A, *d_B;
    cudaMalloc(&d_A, n * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));
    
    // Copy input to device
    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel - one thread per element
    scalarMultiplyKernel<<<1, n>>>(d_A, d_B, k, n);
    
    // Copy result back to host
    cudaMemcpy(h_B, d_B, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print result
    for (int i = 0; i < n; i++) {
        if (i > 0) printf(" ");
        printf("%d", h_B[i]);
    }
    printf("\n");
    
    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void scalarMultiplyKernel()` |
| ✅ Uses parallelism | Each thread processes one element |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, n>>>` launches n threads |
| ✅ Meaningful computation | Performs scalar multiplication on GPU |

---

## CUDA Concepts Covered

### 1. Scalar Multiplication

Each element is multiplied by the same scalar:

```cuda
B[idx] = k * A[idx];

// Examples:
// k=2: [1, 2, 3] → [2, 4, 6]
// k=10: [5, 10, 15] → [50, 100, 150]
```

### 2. Perfect Parallelism

This is an **embarrassingly parallel** problem — each element can be computed independently:

```cuda
scalarMultiplyKernel<<<1, n>>>(d_A, d_B, k, n);
```

```
Thread 0: B[0] = k × A[0]
Thread 1: B[1] = k × A[1]
Thread 2: B[2] = k × A[2]
...
Thread n-1: B[n-1] = k × A[n-1]
```

### 3. Visualization

```
k = 2

Input A:    [1]  [2]  [3]
             │    │    │
             ×2   ×2   ×2
             │    │    │
Output B:   [2]  [4]  [6]

Each thread computes one element independently!
```

### 4. Passing Scalar by Value

Note that k is passed directly to the kernel (not as a pointer):

```cuda
__global__ void scalarMultiplyKernel(int* A, int* B, int k, int n) {
    // k is available directly as a value
    B[idx] = k * A[idx];
}
```

### 5. Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  n = 3, k = 2, A = [1, 2, 3]                             │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (HostToDevice)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   Thread 0: B[0] = 2 × 1 = 2                             │
│   Thread 1: B[1] = 2 × 2 = 4                             │
│   Thread 2: B[2] = 2 × 3 = 6                             │
│                                                          │
│   d_B: [2, 4, 6]                                         │
│                                                          │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (DeviceToHost)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  Output: "2 4 6"                                         │
└──────────────────────────────────────────────────────────┘
```

---

## Alternative Solutions

### Using Multiple Blocks (for Larger Arrays)

```cuda
__global__ void scalarMultiplyKernel(int* A, int* B, int k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        B[idx] = k * A[idx];
    }
}

// Launch with multiple blocks
int threadsPerBlock = 256;
int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
scalarMultiplyKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, k, n);
```

### In-Place Modification

```cuda
__global__ void scalarMultiplyInPlace(int* A, int k, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        A[idx] = k * A[idx];  // Modify in place
    }
}
```

### Sequential (Single Thread)

```cuda
__global__ void scalarMultiplyKernel(int* A, int* B, int k, int n) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < n; i++) {
            B[i] = k * A[i];
        }
    }
}
```

---

## Result Range Consideration

With constraints k ≤ 10,000 and A[i] ≤ 10,000:
- Maximum result: 10,000 × 10,000 = 100,000,000 (100 million)
- This fits within int range (2³¹ - 1 ≈ 2.1 billion) ✓

```cuda
// int is sufficient for this problem
B[idx] = k * A[idx];

// For larger values, use long long:
// long long result = (long long)k * A[idx];
```

---

## Embarrassingly Parallel Problems

Scalar vector multiplication is a classic example of **embarrassingly parallel** computation:

| Property | Description |
|----------|-------------|
| No dependencies | Each output depends only on one input |
| No communication | Threads don't need to share data |
| Perfect scaling | Adding more threads proportionally speeds up |

Other examples:
- Vector addition
- Element-wise operations
- Image filtering (per-pixel operations)

---

## Common Mistakes

### ❌ Missing Bounds Check
```cuda
// Dangerous - may access invalid memory
B[idx] = k * A[idx];

// Safe
if (idx < n) {
    B[idx] = k * A[idx];
}
```

### ❌ Passing k as Pointer Unnecessarily
```cuda
// Overcomplicated
__global__ void kernel(int* A, int* B, int* k, int n) {
    B[idx] = (*k) * A[idx];
}

// Simpler - pass scalar by value
__global__ void kernel(int* A, int* B, int k, int n) {
    B[idx] = k * A[idx];
}
```

### ❌ Wrong Output Format
```cuda
// Wrong - commas instead of spaces
printf("%d,", h_B[i]);

// Correct
if (i > 0) printf(" ");
printf("%d", h_B[i]);
```

---

## Vector Operations Comparison

| Operation | Formula | Parallelism |
|-----------|---------|-------------|
| **Scalar Multiplication** | B[i] = k × A[i] | Perfect |
| Vector Addition | C[i] = A[i] + B[i] | Perfect |
| Dot Product | sum = Σ A[i] × B[i] | Requires reduction |
| Matrix Multiplication | C[i][j] = Σ A[i][k] × B[k][j] | Row-column parallel |

---

## Key Takeaways

1. **Embarrassingly parallel** — no thread dependencies
2. **Pass scalars by value** — simpler than using pointers
3. **Bounds checking** — essential with variable-length arrays
4. **Result range** — ensure data type can hold largest result
5. **One thread per element** — simplest parallel pattern

---

## Practice Exercises

1. Implement **vector addition**: C[i] = A[i] + B[i]
2. Implement **element-wise multiplication**: C[i] = A[i] × B[i]
3. Implement **scalar addition**: B[i] = A[i] + k
4. Handle **negative scalars** and **negative values**

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/39)*
