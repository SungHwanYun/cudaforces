# Product of Two Integers

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given two integers a and b. Print the product a × b.

### Input
The first line contains an integer a.

The second line contains an integer b.

**Constraints:**
- 1 ≤ a, b ≤ 100

### Output
Print the product a × b on the first line.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 13<br>5 | 65 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 10<br>10 | 100 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 7<br>8 | 56 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void productKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*a) * (*b);
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
    productKernel<<<1, 1>>>(d_a, d_b, d_result);
    
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
| ✅ Kernel exists | `__global__ void productKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs multiplication on GPU |

---

## CUDA Concepts Covered

### 1. Multiplication Pattern

Same structure as previous arithmetic operations:

```cuda
// Addition
*result = *a + *b;

// Subtraction
*result = *a - *b;

// Multiplication
*result = (*a) * (*b);
```

### 2. Parentheses for Clarity

While not strictly required, parentheses improve readability:

```cuda
*result = (*a) * (*b);  // Clear: dereference then multiply
*result = *a * *b;      // Also works but harder to read
```

### 3. Result Range Consideration

With inputs 1 ≤ a, b ≤ 100:

```
Minimum product: 1 × 1 = 1
Maximum product: 100 × 100 = 10,000
```

`int` (typically 32 bits) easily holds values up to ~2 billion, so no overflow concern here.

### 4. Data Flow Visualization

```
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  a = 13, b = 5                                           │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (HostToDevice)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   d_a: [13]    d_b: [5]                                  │
│          \      /                                        │
│           \    /                                         │
│            ×   ← Multiplication: 13 × 5 = 65             │
│            │                                             │
│            ▼                                             │
│   d_result: [65]                                         │
│                                                          │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (DeviceToHost)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  result = 65 → printf("65\n")                            │
└──────────────────────────────────────────────────────────┘
```

### 5. Commutativity of Multiplication

Like addition, multiplication is commutative:

```cuda
*result = (*a) * (*b);  // a × b
*result = (*b) * (*a);  // b × a (same result)

// Example: a=13, b=5
// 13 × 5 = 65
// 5 × 13 = 65
```

---

## Alternative Solutions

### Using Array

```cuda
__global__ void productKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        data[2] = data[0] * data[1];
    }
}

int main() {
    int a, b;
    scanf("%d", &a);
    scanf("%d", &b);
    
    int h_data[3] = {a, b, 0};
    
    int* d_data;
    cudaMalloc(&d_data, 3 * sizeof(int));
    cudaMemcpy(d_data, h_data, 3 * sizeof(int), cudaMemcpyHostToDevice);
    
    productKernel<<<1, 1>>>(d_data);
    
    cudaMemcpy(h_data, d_data, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d\n", h_data[2]);
    
    cudaFree(d_data);
    
    return 0;
}
```

### Direct Output in Kernel

```cuda
__global__ void productKernel(int* a, int* b) {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("%d\n", (*a) * (*b));
    }
}
```

---

## Comparison: Basic Arithmetic Operations

| Operation | Symbol | Commutative | Result Range (1-100 inputs) |
|-----------|--------|-------------|----------------------------|
| Addition | `+` | Yes | 2 to 200 |
| Subtraction | `-` | No | -99 to 99 |
| **Multiplication** | `*` | Yes | 1 to 10,000 |
| Division | `/` | No | 0 to 100 |
| Modulo | `%` | No | 0 to 99 |

---

## Common Mistakes

### ❌ Operator Precedence Confusion
```cuda
*result = *a * *b;  // Works but confusing
// Better: use parentheses
*result = (*a) * (*b);
```

### ❌ Overflow for Large Inputs
```cuda
// If inputs could be very large (not in this problem):
int a = 1000000, b = 1000000;
int result = a * b;  // Overflow! Result exceeds int range

// Solution: use long long
long long result = (long long)a * b;
```

### ❌ Using Wrong Operator
```cuda
*result = (*a) + (*b);  // Wrong! This is addition
*result = (*a) * (*b);  // Correct - multiplication
```

---

## Multiplication Use Cases in CUDA

| Use Case | Example |
|----------|---------|
| Scaling | `scaled = value * factor` |
| Area calculation | `area = width * height` |
| Index calculation | `idx = row * cols + col` |
| Weighted sum | `sum = weight * value` |

The global thread index formula is a key example:
```cuda
int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
//                        ↑ multiplication
```

---

## Key Takeaways

1. **Same pattern** as addition and subtraction
2. **Commutative** — order doesn't affect result
3. **Larger results** — product can be much larger than inputs
4. **Use parentheses** for clarity with pointer dereferencing
5. **Foundation for indexing** — multiplication crucial for thread indexing

---

## Practice Exercises

1. Compute the **square** of a single integer (a × a)
2. Compute **a × b × c** for three integers
3. Compute the **area** of a rectangle given width and height
4. Consider overflow: what if inputs could be up to 10^9?

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/23)*
