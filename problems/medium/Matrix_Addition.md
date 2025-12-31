# Matrix Addition

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Medium | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given two n × n integer matrices A and B. Compute the element-wise sum of the two matrices and output the resulting matrix C, where:

$$C[i][j] = A[i][j] + B[i][j]$$

for all 0 ≤ i, j < n.

### Input
The first line contains an integer n (1 ≤ n ≤ 10), the size of the matrices.

The next n lines contain n space-separated integers each, representing matrix A (1 ≤ A[i][j] ≤ 100).

The next n lines contain n space-separated integers each, representing matrix B (1 ≤ B[i][j] ≤ 100).

### Output
Output n lines, each containing n space-separated integers representing the resulting matrix C.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 2<br>1 2<br>3 4<br>5 6<br>7 8 | 6 8<br>10 12 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 3<br>1 2 3<br>4 5 6<br>7 8 9<br>9 8 7<br>6 5 4<br>3 2 1 | 10 10 10<br>10 10 10<br>10 10 10 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 1<br>100<br>100 | 200 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void matrixAddKernel(int* A, int* B, int* C, int n) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    if (row < n && col < n) {
        int idx = row * n + col;
        C[idx] = A[idx] + B[idx];
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
    matrixAddKernel<<<1, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
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
| ✅ Kernel exists | `__global__ void matrixAddKernel()` |
| ✅ Uses parallelism | 2D thread block with `threadIdx.x` and `threadIdx.y` |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, dim3(n,n)>>>` launches n×n threads |
| ✅ Meaningful computation | Performs element-wise addition on GPU |

---

## CUDA Concepts Covered

### 1. Element-wise Addition

Unlike matrix multiplication, addition is simple — just add corresponding elements:

```cuda
C[idx] = A[idx] + B[idx];
```

No summation loop needed!

### 2. 2D Thread Indexing

Same pattern as matrix multiplication:

```cuda
int row = threadIdx.y;
int col = threadIdx.x;
int idx = row * n + col;
```

### 3. Visualization (Example 1)

```
Matrix A:        Matrix B:        Matrix C:
┌─────────┐      ┌─────────┐      ┌─────────┐
│ 1  2    │  +   │ 5  6    │  =   │ 6   8   │
│ 3  4    │      │ 7  8    │      │ 10  12  │
└─────────┘      └─────────┘      └─────────┘

C[0][0] = 1 + 5 = 6
C[0][1] = 2 + 6 = 8
C[1][0] = 3 + 7 = 10
C[1][1] = 4 + 8 = 12
```

### 4. Complement Matrices (Example 2)

```
Matrix A:           Matrix B:           Matrix C:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 1  2  3     │     │ 9  8  7     │     │ 10 10 10    │
│ 4  5  6     │  +  │ 6  5  4     │  =  │ 10 10 10    │
│ 7  8  9     │     │ 3  2  1     │     │ 10 10 10    │
└─────────────┘     └─────────────┘     └─────────────┘

A[i][j] + B[i][j] = 10 for all i, j
```

### 5. Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   Thread (0,0): C[0] = A[0] + B[0]                       │
│   Thread (1,0): C[1] = A[1] + B[1]                       │
│   Thread (0,1): C[2] = A[2] + B[2]                       │
│   Thread (1,1): C[3] = A[3] + B[3]                       │
│                                                          │
│   All threads execute simultaneously!                    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Alternative Solutions

### Using 1D Thread Block

```cuda
__global__ void matrixAddKernel(int* A, int* B, int* C, int n) {
    int idx = threadIdx.x;
    if (idx < n * n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Launch with n*n threads in 1D
matrixAddKernel<<<1, n*n>>>(d_A, d_B, d_C, n);
```

### Using Multiple Blocks

```cuda
__global__ void matrixAddKernel(int* A, int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        int idx = row * n + col;
        C[idx] = A[idx] + B[idx];
    }
}

// Launch with 2D grid
dim3 threadsPerBlock(16, 16);
dim3 numBlocks((n + 15) / 16, (n + 15) / 16);
matrixAddKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
```

### In-Place Addition

```cuda
__global__ void matrixAddInPlace(int* A, int* B, int n) {
    int idx = threadIdx.y * n + threadIdx.x;
    if (threadIdx.y < n && threadIdx.x < n) {
        A[idx] = A[idx] + B[idx];  // Result stored in A
    }
}
```

---

## Matrix Addition vs Multiplication

| Aspect | Addition | Multiplication |
|--------|----------|----------------|
| Formula | C[i][j] = A[i][j] + B[i][j] | C[i][j] = Σ A[i][k] × B[k][j] |
| Complexity | O(1) per element | O(n) per element |
| Dependencies | None | Needs entire row/column |
| Parallelism | Perfect | Good |

Matrix addition is **simpler** and **more parallel** than multiplication!

---

## Common Mistakes

### ❌ Using Wrong Index Formula
```cuda
// Wrong - column-major order
int idx = col * n + row;

// Correct - row-major order
int idx = row * n + col;
```

### ❌ Confusing with Multiplication
```cuda
// Wrong - multiplication formula
C[idx] = A[idx] * B[idx];

// Correct - addition
C[idx] = A[idx] + B[idx];
```

### ❌ Missing Bounds Check
```cuda
// Should check bounds for safety
if (row < n && col < n) {
    // ... computation
}
```

---

## Embarrassingly Parallel

Matrix addition is **embarrassingly parallel**:

```
Each output C[i][j] depends only on:
- A[i][j]
- B[i][j]

No dependencies between elements!
All n² additions can happen simultaneously.
```

---

## Key Takeaways

1. **Element-wise operation** — each element computed independently
2. **2D thread block** — natural mapping to 2D matrices
3. **Simpler than multiplication** — no summation loop
4. **Perfect parallelism** — no thread dependencies
5. **Same indexing pattern** — `row * n + col` for row-major

---

## Practice Exercises

1. Implement **matrix subtraction**: C[i][j] = A[i][j] - B[i][j]
2. Implement **element-wise multiplication**: C[i][j] = A[i][j] × B[i][j]
3. Implement **scalar matrix addition**: C[i][j] = A[i][j] + k
4. Add **three matrices**: D[i][j] = A[i][j] + B[i][j] + C[i][j]

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/40)*
