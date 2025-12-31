# Vector Addition

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Medium | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given two integer vectors A and B, each of size n. Compute the element-wise sum of the two vectors and output the resulting vector C, where:

$$C[i] = A[i] + B[i]$$

for all 0 ≤ i < n.

### Input
The first line contains an integer n (1 ≤ n ≤ 100), the size of the vectors.

The second line contains n space-separated integers representing vector A (-10,000 ≤ A[i] ≤ 10,000).

The third line contains n space-separated integers representing vector B (-10,000 ≤ B[i] ≤ 10,000).

### Output
Output n space-separated integers representing the resulting vector C.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 3<br>1 2 3<br>4 5 6 | 5 7 9 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 5<br>-1 0 3 -2 5<br>2 -3 1 4 -1 | 1 -3 4 2 4 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 1<br>10<br>-10 | 0 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void vectorAddKernel(int* A, int* B, int* C, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int n;
    scanf("%d", &n);
    
    int* h_A = (int*)malloc(n * sizeof(int));
    int* h_B = (int*)malloc(n * sizeof(int));
    int* h_C = (int*)malloc(n * sizeof(int));
    
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_A[i]);
    }
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_B[i]);
    }
    
    // Device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));
    cudaMalloc(&d_C, n * sizeof(int));
    
    // Copy inputs to device
    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel - one thread per element
    vectorAddKernel<<<1, n>>>(d_A, d_B, d_C, n);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print result
    for (int i = 0; i < n; i++) {
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
| ✅ Kernel exists | `__global__ void vectorAddKernel()` |
| ✅ Uses parallelism | Each thread processes one element |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, n>>>` launches n threads |
| ✅ Meaningful computation | Performs element-wise addition on GPU |

---

## CUDA Concepts Covered

### 1. Element-wise Vector Addition

The simplest parallel operation — each element is computed independently:

```cuda
C[idx] = A[idx] + B[idx];
```

### 2. Visualization (Example 1)

```
Vector A:    [1]    [2]    [3]
              +      +      +
Vector B:    [4]    [5]    [6]
              =      =      =
Vector C:    [5]    [7]    [9]

Thread 0: C[0] = 1 + 4 = 5
Thread 1: C[1] = 2 + 5 = 7
Thread 2: C[2] = 3 + 6 = 9
```

### 3. Perfect Parallelism

Vector addition is the **"Hello World" of GPU computing** — perfectly parallel:

```
┌─────────────────────────────────────┐
│            GPU Kernel               │
│                                     │
│  Thread 0: C[0] = A[0] + B[0]       │
│  Thread 1: C[1] = A[1] + B[1]       │
│  Thread 2: C[2] = A[2] + B[2]       │
│  ...                                │
│  Thread n-1: C[n-1] = A[n-1] + B[n-1]│
│                                     │
│  All execute simultaneously!        │
└─────────────────────────────────────┘
```

### 4. Negative Numbers (Example 2)

```
A:  [-1]   [0]   [3]   [-2]   [5]
     +      +     +      +     +
B:   [2]  [-3]   [1]    [4]  [-1]
     =      =     =      =     =
C:   [1]  [-3]   [4]    [2]   [4]
```

### 5. Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  A = [1, 2, 3]                                           │
│  B = [4, 5, 6]                                           │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (HostToDevice)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   d_A: [1, 2, 3]                                         │
│   d_B: [4, 5, 6]                                         │
│                                                          │
│   Thread 0: d_C[0] = 1 + 4 = 5                           │
│   Thread 1: d_C[1] = 2 + 5 = 7                           │
│   Thread 2: d_C[2] = 3 + 6 = 9                           │
│                                                          │
│   d_C: [5, 7, 9]                                         │
│                                                          │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (DeviceToHost)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  Output: "5 7 9"                                         │
└──────────────────────────────────────────────────────────┘
```

---

## Alternative Solutions

### Using Multiple Blocks

```cuda
__global__ void vectorAddKernel(int* A, int* B, int* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Launch with multiple blocks for larger arrays
int threadsPerBlock = 256;
int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
vectorAddKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
```

### In-Place Addition

```cuda
__global__ void vectorAddInPlace(int* A, int* B, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        A[idx] = A[idx] + B[idx];  // Result stored in A
    }
}
```

### Sequential (Single Thread)

```cuda
__global__ void vectorAddKernel(int* A, int* B, int* C, int n) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < n; i++) {
            C[i] = A[i] + B[i];
        }
    }
}
```

---

## Comparison with Related Operations

| Operation | Formula | Description |
|-----------|---------|-------------|
| **Vector Addition** | C[i] = A[i] + B[i] | Element-wise sum |
| Vector Subtraction | C[i] = A[i] - B[i] | Element-wise difference |
| Scalar Multiplication | C[i] = k × A[i] | Scale by constant |
| Hadamard Product | C[i] = A[i] × B[i] | Element-wise multiply |
| Dot Product | Σ A[i] × B[i] | Single scalar result |

---

## Vector Addition as "Hello World"

Vector addition is traditionally the first CUDA program because:

1. **Simple formula**: Just one addition per element
2. **Perfect parallelism**: No dependencies between elements
3. **Clear mapping**: One thread per element
4. **Demonstrates core concepts**: Memory allocation, transfer, kernel launch

```
CPU "Hello World":  printf("Hello World!\n");
GPU "Hello World":  C[i] = A[i] + B[i];
```

---

## Common Mistakes

### ❌ Missing Bounds Check
```cuda
// Dangerous
C[idx] = A[idx] + B[idx];

// Safe
if (idx < n) {
    C[idx] = A[idx] + B[idx];
}
```

### ❌ Forgetting to Copy B
```cuda
// Wrong - only copied A
cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
// Missing: cudaMemcpy(d_B, h_B, ...)
```

### ❌ Wrong Output Format
```cuda
// Wrong - no spaces between numbers
for (int i = 0; i < n; i++) printf("%d", h_C[i]);

// Correct
for (int i = 0; i < n; i++) {
    if (i > 0) printf(" ");
    printf("%d", h_C[i]);
}
```

---

## Embarrassingly Parallel

Vector addition is **embarrassingly parallel**:

| Property | Vector Addition |
|----------|-----------------|
| Dependencies | None |
| Communication | None needed |
| Synchronization | None needed |
| Speedup | Linear with threads |

Every element can be computed completely independently!

---

## Key Takeaways

1. **GPU "Hello World"** — the classic first CUDA program
2. **Perfect parallelism** — each element computed independently
3. **Simple formula** — C[i] = A[i] + B[i]
4. **Foundation** — same pattern for subtraction, multiplication, etc.
5. **Bounds checking** — always verify idx < n

---

## Practice Exercises

1. Implement **vector subtraction**: C[i] = A[i] - B[i]
2. Implement **three-vector addition**: D[i] = A[i] + B[i] + C[i]
3. Implement **weighted sum**: C[i] = αA[i] + βB[i]
4. Implement **vector magnitude**: |A| = √(Σ A[i]²)

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/4)*
