# 1D Convolution

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Medium | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given an input array A of size n and a kernel B of size m. Compute the 1D convolution (valid mode) and output the resulting array C of size n - m + 1, where:

$$C[i] = \sum_{k=0}^{m-1} A[i + k] \times B[k]$$

for all 0 ≤ i < n - m + 1 and 0 ≤ k < m.

### Input
The first line contains two integers n and m (1 ≤ n ≤ 100, 1 ≤ m ≤ 100, m ≤ n), the size of the input array and the kernel.

The second line contains n space-separated integers representing array A (1 ≤ A[i] ≤ 100).

The third line contains m space-separated integers representing kernel B (1 ≤ B[i] ≤ 100).

### Output
Output n - m + 1 space-separated integers representing the resulting array C.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 5 3<br>1 2 3 4 5<br>1 1 1 | 6 9 12 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 7 2<br>1 3 5 7 9 11 13<br>1 2 | 7 13 19 25 31 37 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 4 4<br>2 4 6 8<br>1 1 1 1 | 20 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void convolution1DKernel(int* A, int* B, int* C, int n, int m) {
    int idx = threadIdx.x;
    int outputSize = n - m + 1;
    
    if (idx < outputSize) {
        int sum = 0;
        for (int k = 0; k < m; k++) {
            sum += A[idx + k] * B[k];
        }
        C[idx] = sum;
    }
}

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    
    int outputSize = n - m + 1;
    
    int* h_A = (int*)malloc(n * sizeof(int));
    int* h_B = (int*)malloc(m * sizeof(int));
    int* h_C = (int*)malloc(outputSize * sizeof(int));
    
    // Read input array A
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_A[i]);
    }
    
    // Read kernel B
    for (int i = 0; i < m; i++) {
        scanf("%d", &h_B[i]);
    }
    
    // Device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * sizeof(int));
    cudaMalloc(&d_B, m * sizeof(int));
    cudaMalloc(&d_C, outputSize * sizeof(int));
    
    // Copy inputs to device
    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, m * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel - one thread per output element
    convolution1DKernel<<<1, outputSize>>>(d_A, d_B, d_C, n, m);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, outputSize * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print result
    for (int i = 0; i < outputSize; i++) {
        if (i > 0) printf(" ");
        printf("%d", h_C[i]);
    }
    printf("\n");
    
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void convolution1DKernel()` |
| ✅ Uses parallelism | Each thread computes one output element |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, outputSize>>>` launches threads |
| ✅ Meaningful computation | Performs 1D convolution on GPU |

---

## CUDA Concepts Covered

### 1. What is Convolution?

Convolution slides a kernel (filter) across the input, computing weighted sums at each position:

```cuda
int sum = 0;
for (int k = 0; k < m; k++) {
    sum += A[idx + k] * B[k];
}
C[idx] = sum;
```

### 2. Visualization (Example 1)

```
Input A:  [1]  [2]  [3]  [4]  [5]    (n = 5)
Kernel B: [1]  [1]  [1]              (m = 3)

Position 0:
   [1]  [2]  [3]  [4]  [5]
   [1]  [1]  [1]
   1×1 + 2×1 + 3×1 = 6              → C[0] = 6

Position 1:
   [1]  [2]  [3]  [4]  [5]
        [1]  [1]  [1]
        2×1 + 3×1 + 4×1 = 9         → C[1] = 9

Position 2:
   [1]  [2]  [3]  [4]  [5]
             [1]  [1]  [1]
             3×1 + 4×1 + 5×1 = 12   → C[2] = 12

Output C: [6]  [9]  [12]            (n - m + 1 = 3)
```

### 3. Output Size

In "valid" mode (no padding), output size is:
```
outputSize = n - m + 1
```

Examples:
- n=5, m=3 → output = 3
- n=7, m=2 → output = 6
- n=4, m=4 → output = 1

### 4. Sliding Window

Each thread handles one window position:

```
Thread 0: A[0..m-1] × B[0..m-1] → C[0]
Thread 1: A[1..m]   × B[0..m-1] → C[1]
Thread 2: A[2..m+1] × B[0..m-1] → C[2]
...
```

### 5. Weighted Sum (Example 2)

```
Kernel [1, 2] applies different weights:

A = [1, 3, 5, 7, 9, 11, 13]
B = [1, 2]

C[0] = 1×1 + 3×2 = 1 + 6 = 7
C[1] = 3×1 + 5×2 = 3 + 10 = 13
C[2] = 5×1 + 7×2 = 5 + 14 = 19
...
```

---

## Alternative Solutions

### Using Multiple Blocks

```cuda
__global__ void convolution1DKernel(int* A, int* B, int* C, int n, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int outputSize = n - m + 1;
    
    if (idx < outputSize) {
        int sum = 0;
        for (int k = 0; k < m; k++) {
            sum += A[idx + k] * B[k];
        }
        C[idx] = sum;
    }
}

// Launch
int threadsPerBlock = 256;
int numBlocks = (outputSize + threadsPerBlock - 1) / threadsPerBlock;
convolution1DKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n, m);
```

### Using Shared Memory for Kernel

```cuda
__global__ void convolution1DKernel(int* A, int* B, int* C, int n, int m) {
    __shared__ int sharedB[100];  // Shared kernel
    
    int idx = threadIdx.x;
    int outputSize = n - m + 1;
    
    // Load kernel to shared memory
    if (idx < m) {
        sharedB[idx] = B[idx];
    }
    __syncthreads();
    
    if (idx < outputSize) {
        int sum = 0;
        for (int k = 0; k < m; k++) {
            sum += A[idx + k] * sharedB[k];
        }
        C[idx] = sum;
    }
}
```

### Sequential (Single Thread)

```cuda
__global__ void convolution1DKernel(int* A, int* B, int* C, int n, int m) {
    if (threadIdx.x == 0) {
        int outputSize = n - m + 1;
        for (int i = 0; i < outputSize; i++) {
            int sum = 0;
            for (int k = 0; k < m; k++) {
                sum += A[i + k] * B[k];
            }
            C[i] = sum;
        }
    }
}
```

---

## Convolution Modes

| Mode | Output Size | Description |
|------|-------------|-------------|
| **Valid** | n - m + 1 | No padding (this problem) |
| Same | n | Output same size as input |
| Full | n + m - 1 | All overlapping positions |

---

## Common Applications

1D convolution is used in:
- **Signal Processing**: Audio filtering, smoothing
- **Deep Learning**: CNN layers for 1D data (time series, text)
- **Image Processing**: Edge detection (2D extension)
- **Moving Average**: Kernel [1/m, 1/m, ..., 1/m]

### Common Kernels

| Kernel | Effect | Example |
|--------|--------|---------|
| [1, 1, 1] | Sum / Blur | Moving sum |
| [1, 0, -1] | Derivative | Edge detection |
| [1, 2, 1] | Gaussian-like | Smoothing |
| [1, -2, 1] | Second derivative | Curvature |

---

## Common Mistakes

### ❌ Wrong Output Size
```cuda
// Wrong - using n instead of n-m+1
int outputSize = n;

// Correct
int outputSize = n - m + 1;
```

### ❌ Array Index Out of Bounds
```cuda
// Wrong - accessing A[idx + k] when idx + k >= n
sum += A[idx + k] * B[k];

// Safe - bounds check ensures idx < outputSize
if (idx < outputSize) { ... }
```

### ❌ Wrong Summation Index
```cuda
// Wrong - using idx instead of k for B
sum += A[idx + k] * B[idx];

// Correct
sum += A[idx + k] * B[k];
```

---

## Comparison: Convolution vs Matrix Multiply

| Aspect | 1D Convolution | Matrix Multiply |
|--------|----------------|-----------------|
| Input | 1D array + kernel | 2D matrices |
| Output size | n - m + 1 | n × n |
| Inner loop | Over kernel (m) | Over k (n) |
| Pattern | Sliding window | Row × Column |

Both involve dot products, but with different access patterns!

---

## Key Takeaways

1. **Sliding window** — kernel slides across input
2. **Valid mode** — output size is n - m + 1
3. **Dot product at each position** — weighted sum of window
4. **Parallelizable** — each output element is independent
5. **Foundation for CNNs** — extends to 2D for images

---

## Practice Exercises

1. Implement **2D convolution** for image filtering
2. Implement **"same" mode** with zero padding
3. Use **shared memory** to cache the kernel
4. Implement **pooling** (max or average of windows)

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/42)*
