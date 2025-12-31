# Vector ReLU

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Medium | 128 MB | 1 s | MenOfPassion |

## Problem Description

ReLU (Rectified Linear Unit) is one of the most widely used activation functions in deep learning. Given a vector **A** of **n** integers, compute the ReLU of each element.

The ReLU function is defined as:

```
ReLU(x) = max(0, x)
```

### Input
The first line contains an integer **n** (1 ≤ n ≤ 100), the size of vector **A**.

The second line contains **n** space-separated integers **A[i]** (-100 ≤ A[i] ≤ 100).

### Output
Print **n** space-separated integers, where each value is the ReLU of the corresponding element in **A**.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 5<br>-3 7 0 -1 4 | 0 7 0 0 4 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 4<br>-50 -25 -10 -1 | 0 0 0 0 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 6<br>10 0 20 0 30 0 | 10 0 20 0 30 0 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void reluKernel(int* A, int* B, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        B[idx] = (A[idx] > 0) ? A[idx] : 0;
    }
}

int main() {
    int n;
    scanf("%d", &n);
    
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
    reluKernel<<<1, n>>>(d_A, d_B, n);
    
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
| ✅ Kernel exists | `__global__ void reluKernel()` |
| ✅ Uses parallelism | Each thread processes one element |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, n>>>` launches n threads |
| ✅ Meaningful computation | Performs ReLU activation on GPU |

---

## CUDA Concepts Covered

### 1. ReLU Function

ReLU is the simplest non-linear activation function:

```cuda
// Using ternary operator
B[idx] = (A[idx] > 0) ? A[idx] : 0;

// Or using max
B[idx] = max(0, A[idx]);
```

```
ReLU(x) = { x  if x > 0
          { 0  if x ≤ 0
```

### 2. Visualization

```
Input:   -3    7    0   -1    4
          │    │    │    │    │
        ReLU  ReLU ReLU ReLU ReLU
          │    │    │    │    │
Output:   0    7    0    0    4

ReLU Graph:
        y │
          │      ╱
          │     ╱
          │    ╱
        ──┼───╱────── x
          │  ╱
          │ ╱ (0 for all x < 0)
```

### 3. Perfect Parallelism

ReLU is **embarrassingly parallel** — each element can be computed independently:

```
Thread 0: B[0] = ReLU(A[0]) = max(0, -3) = 0
Thread 1: B[1] = ReLU(A[1]) = max(0, 7) = 7
Thread 2: B[2] = ReLU(A[2]) = max(0, 0) = 0
Thread 3: B[3] = ReLU(A[3]) = max(0, -1) = 0
Thread 4: B[4] = ReLU(A[4]) = max(0, 4) = 4
```

### 4. Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  A = [-3, 7, 0, -1, 4]                                   │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (HostToDevice)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   Thread 0: max(0, -3) = 0                               │
│   Thread 1: max(0, 7) = 7                                │
│   Thread 2: max(0, 0) = 0                                │
│   Thread 3: max(0, -1) = 0                               │
│   Thread 4: max(0, 4) = 4                                │
│                                                          │
│   d_B: [0, 7, 0, 0, 4]                                   │
│                                                          │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (DeviceToHost)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  Output: "0 7 0 0 4"                                     │
└──────────────────────────────────────────────────────────┘
```

### 5. Why ReLU in Deep Learning?

ReLU is popular because:
- **Simple**: Just max(0, x)
- **Fast**: Single comparison
- **Effective**: Helps with vanishing gradient problem
- **Sparse**: Many neurons output 0

---

## Alternative Solutions

### Using max() Function

```cuda
__global__ void reluKernel(int* A, int* B, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        B[idx] = max(0, A[idx]);
    }
}
```

### Using If Statement

```cuda
__global__ void reluKernel(int* A, int* B, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        if (A[idx] > 0) {
            B[idx] = A[idx];
        } else {
            B[idx] = 0;
        }
    }
}
```

### In-Place Modification

```cuda
__global__ void reluInPlace(int* A, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        if (A[idx] < 0) {
            A[idx] = 0;
        }
    }
}
```

### Using Multiplication Trick

```cuda
__global__ void reluKernel(int* A, int* B, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        // (A[idx] > 0) evaluates to 1 or 0
        B[idx] = A[idx] * (A[idx] > 0);
    }
}
```

---

## Common Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| **ReLU** | max(0, x) | [0, ∞) | Most common |
| Leaky ReLU | max(0.01x, x) | (-∞, ∞) | Avoids dead neurons |
| Sigmoid | 1/(1+e⁻ˣ) | (0, 1) | Binary classification |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | Centered output |
| Softmax | eˣⁱ/Σeˣʲ | (0, 1) | Multi-class |

---

## ReLU Properties

| Property | Value |
|----------|-------|
| Output range | [0, ∞) |
| Derivative | 1 if x > 0, 0 if x ≤ 0 |
| Continuity | Continuous |
| Differentiability | Not differentiable at x = 0 |
| Monotonicity | Non-decreasing |

---

## Common Mistakes

### ❌ Using >= Instead of >
```cuda
// Both are technically correct for integers
// but convention uses > 0
B[idx] = (A[idx] >= 0) ? A[idx] : 0;  // Includes 0
B[idx] = (A[idx] > 0) ? A[idx] : 0;   // Standard ReLU
// For integers: ReLU(0) = 0 either way
```

### ❌ Forgetting Bounds Check
```cuda
// Dangerous without bounds check
B[idx] = max(0, A[idx]);

// Safe
if (idx < n) {
    B[idx] = max(0, A[idx]);
}
```

### ❌ Wrong Comparison Order
```cuda
// Wrong for max() style
B[idx] = max(A[idx], 0);  // Reversed but still works

// Standard convention
B[idx] = max(0, A[idx]);  // 0 first
```

---

## Neural Network Context

In a neural network, ReLU is applied after linear transformations:

```
Layer computation:
1. Linear: z = Wx + b
2. Activation: a = ReLU(z)

Example for single neuron:
Input: x = [1, 2, 3]
Weights: w = [0.5, -0.3, 0.2]
Bias: b = -1

z = 0.5×1 + (-0.3)×2 + 0.2×3 + (-1)
  = 0.5 - 0.6 + 0.6 - 1
  = -0.5

a = ReLU(-0.5) = 0
```

---

## Key Takeaways

1. **ReLU = max(0, x)** — simple and effective
2. **Element-wise operation** — perfect parallelism
3. **Zero for negatives** — creates sparse activations
4. **Foundation of deep learning** — most common activation
5. **GPU-friendly** — simple operation, massive parallelism

---

## Practice Exercises

1. Implement **Leaky ReLU**: max(0.01x, x)
2. Implement **Sigmoid**: 1/(1+e⁻ˣ)
3. Apply ReLU to a **2D matrix**
4. Implement **parametric ReLU** with custom slope

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/47)*
