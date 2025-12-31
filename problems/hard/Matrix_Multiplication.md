# Matrix Multiplication

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Medium | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given two n × n integer matrices A and B. Compute the matrix product and output the resulting matrix C, where:

$$C[i][j] = \sum_{k=0}^{n-1} A[i][k] \times B[k][j]$$

for all 0 ≤ i, j < n and 0 ≤ k < n.

### Input
The first line contains an integer n (1 ≤ n ≤ 10), the size of the matrices.

The next n lines contain n space-separated integers each, representing matrix A (0 ≤ A[i][j] ≤ 100).

The next n lines contain n space-separated integers each, representing matrix B (0 ≤ B[i][j] ≤ 100).

### Output
Output n lines, each containing n space-separated integers representing the resulting matrix C.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 2<br>1 2<br>3 4<br>5 6<br>7 8 | 19 22<br>43 50 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 3<br>1 0 0<br>0 1 0<br>0 0 1<br>1 2 3<br>4 5 6<br>7 8 9 | 1 2 3<br>4 5 6<br>7 8 9 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 1<br>10<br>10 | 100 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void matrixMultiplyKernel(int* A, int* B, int* C, int n) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    int n;
    scanf("%d", &n);
    
    int* h_A = (int*)malloc(n * n * sizeof(int));
    int* h_B = (int*)malloc(n * n * sizeof(int));
    int* h_C = (int*)malloc(n * n * sizeof(int));
    
    // Read matrix A
    for (int i = 0; i < n * n; i++) {
        scanf("%d", &h_A[i]);
    }
    
    // Read matrix B
    for (int i = 0; i < n * n; i++) {
        scanf("%d", &h_B[i]);
    }
    
    // Device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * n * sizeof(int));
    cudaMalloc(&d_B, n * n * sizeof(int));
    cudaMalloc(&d_C, n * n * sizeof(int));
    
    // Copy inputs to device
    cudaMemcpy(d_A, h_A, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel with 2D thread block
    dim3 threadsPerBlock(n, n);
    matrixMultiplyKernel<<<1, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print result
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j > 0) printf(" ");
            printf("%d", h_C[i * n + j]);
        }
        printf("\n");
    }
    
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
| ✅ Kernel exists | `__global__ void matrixMultiplyKernel()` |
| ✅ Uses parallelism | 2D thread block with `threadIdx.x` and `threadIdx.y` |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, dim3(n,n)>>>` launches n×n threads |
| ✅ Meaningful computation | Performs matrix multiplication on GPU |

---

## CUDA Concepts Covered

### 1. Matrix Multiplication Formula

Each element C[i][j] is the dot product of row i of A and column j of B:

```cuda
C[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j] + ... + A[i][n-1]*B[n-1][j]
```

```cuda
int sum = 0;
for (int k = 0; k < n; k++) {
    sum += A[row * n + k] * B[k * n + col];
}
C[row * n + col] = sum;
```

### 2. 2D Thread Indexing

Use `threadIdx.x` and `threadIdx.y` for column and row:

```cuda
int row = threadIdx.y;
int col = threadIdx.x;
```

```
Thread Block (n=3):
┌─────────────────────────────┐
│ (0,0)   (1,0)   (2,0)       │  threadIdx.y = 0
│ (0,1)   (1,1)   (2,1)       │  threadIdx.y = 1
│ (0,2)   (1,2)   (2,2)       │  threadIdx.y = 2
└─────────────────────────────┘
  col=0   col=1   col=2
  threadIdx.x
```

### 3. dim3 for 2D Launch

```cuda
dim3 threadsPerBlock(n, n);  // (x, y) dimensions
matrixMultiplyKernel<<<1, threadsPerBlock>>>(d_A, d_B, d_C, n);
```

This creates a 2D grid of n × n threads.

### 4. Row-Major Memory Layout

2D matrices are stored in 1D arrays in row-major order:

```
Matrix:           Memory:
┌─────────┐       ┌───┬───┬───┬───┐
│ 1  2    │  →    │ 1 │ 2 │ 3 │ 4 │
│ 3  4    │       └───┴───┴───┴───┘
└─────────┘        [0] [1] [2] [3]

Index formula: A[row][col] = A[row * n + col]
```

### 5. Visualization (Example 1)

```
Matrix A:        Matrix B:        Matrix C:
┌─────────┐      ┌─────────┐      ┌─────────┐
│ 1  2    │  ×   │ 5  6    │  =   │ 19  22  │
│ 3  4    │      │ 7  8    │      │ 43  50  │
└─────────┘      └─────────┘      └─────────┘

C[0][0] = 1×5 + 2×7 = 5 + 14 = 19
C[0][1] = 1×6 + 2×8 = 6 + 16 = 22
C[1][0] = 3×5 + 4×7 = 15 + 28 = 43
C[1][1] = 3×6 + 4×8 = 18 + 32 = 50
```

---

## Alternative Solutions

### Using 1D Thread Block

```cuda
__global__ void matrixMultiplyKernel(int* A, int* B, int* C, int n) {
    int idx = threadIdx.x;
    int row = idx / n;
    int col = idx % n;
    
    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Launch with n*n threads
matrixMultiplyKernel<<<1, n*n>>>(d_A, d_B, d_C, n);
```

### Using Multiple Blocks (for Larger Matrices)

```cuda
__global__ void matrixMultiplyKernel(int* A, int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Launch with multiple blocks
dim3 threadsPerBlock(16, 16);
dim3 numBlocks((n + 15) / 16, (n + 15) / 16);
matrixMultiplyKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
```

### Sequential (Single Thread)

```cuda
__global__ void matrixMultiplyKernel(int* A, int* B, int* C, int n) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += A[i * n + k] * B[k * n + j];
                }
                C[i * n + j] = sum;
            }
        }
    }
}
```

---

## Parallel vs Sequential

| Approach | Threads | Work per Thread | Time Complexity |
|----------|---------|-----------------|-----------------|
| Parallel (2D) | n² | O(n) loop | O(n) per thread |
| Sequential | 1 | O(n³) loops | O(n³) |

The parallel approach is n² times faster theoretically!

---

## Thread-to-Element Mapping

```
Thread (tx, ty)  →  Computes C[ty][tx]

          col (threadIdx.x)
          0     1     2
        ┌─────┬─────┬─────┐
row   0 │(0,0)│(1,0)│(2,0)│  → C[0][0], C[0][1], C[0][2]
(ty)  1 │(0,1)│(1,1)│(2,1)│  → C[1][0], C[1][1], C[1][2]
      2 │(0,2)│(1,2)│(2,2)│  → C[2][0], C[2][1], C[2][2]
        └─────┴─────┴─────┘
```

---

## Identity Matrix Property (Example 2)

```
I × B = B

Where I is the identity matrix:
┌─────────────┐
│ 1  0  0     │
│ 0  1  0     │
│ 0  0  1     │
└─────────────┘

I[i][j] = 1 if i == j, else 0
```

This is a useful test case!

---

## Common Mistakes

### ❌ Wrong Index Calculation
```cuda
// Wrong - swapped row and column
C[col * n + row] = sum;

// Correct
C[row * n + col] = sum;
```

### ❌ Wrong Loop Variable
```cuda
// Wrong - using wrong index for A or B
sum += A[row * n + col] * B[k * n + col];

// Correct
sum += A[row * n + k] * B[k * n + col];
```

### ❌ Missing Bounds Check
```cuda
// Should check bounds for variable-size matrices
if (row < n && col < n) {
    // ... computation
}
```

### ❌ Uninitialized Sum
```cuda
int sum;  // Uninitialized - may have garbage value
int sum = 0;  // Correct
```

---

## Output Formatting

```cuda
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        if (j > 0) printf(" ");  // Space between elements
        printf("%d", h_C[i * n + j]);
    }
    printf("\n");  // Newline after each row
}
```

---

## Key Takeaways

1. **2D thread indexing** — `threadIdx.x` for column, `threadIdx.y` for row
2. **Row-major layout** — `A[i][j]` = `A[i * n + j]`
3. **dim3 launch** — `dim3(x, y)` for 2D thread blocks
4. **Parallel computation** — each thread computes one element
5. **O(n) per thread** — dot product requires n multiplications

---

## Practice Exercises

1. Implement **matrix addition**: C[i][j] = A[i][j] + B[i][j]
2. Implement **matrix transpose**: C[i][j] = A[j][i]
3. Use **shared memory** for tile-based matrix multiplication
4. Handle **non-square matrices** (m × n) × (n × p)

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/25)*
