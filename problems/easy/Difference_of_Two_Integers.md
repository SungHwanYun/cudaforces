# Difference of Two Integers

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given two integers a and b. Print the difference a - b.

### Input
The first line contains an integer a.

The second line contains an integer b.

**Constraints:**
- 1 ≤ a, b ≤ 100

### Output
Print the difference a - b on the first line.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 13<br>7 | 6 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 100<br>42 | 58 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 50<br>50 | 0 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void diffKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = *a - *b;
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
    diffKernel<<<1, 1>>>(d_a, d_b, d_result);
    
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
| ✅ Kernel exists | `__global__ void diffKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs subtraction on GPU |

---

## CUDA Concepts Covered

### 1. Subtraction Pattern

Same structure as addition, different operator:

```cuda
// Addition (previous problem)
*result = *a + *b;

// Subtraction (this problem)
*result = *a - *b;
```

### 2. Order Matters in Subtraction

Unlike addition, subtraction is **not commutative**:

```cuda
*result = *a - *b;  // a - b
*result = *b - *a;  // b - a (different result!)

// Example: a=13, b=7
// a - b = 6
// b - a = -6
```

### 3. Handling Negative Results

Even with positive inputs (1 ≤ a, b ≤ 100), result can be negative:

```
Input: a=50, b=50 → Output: 0
Input: a=7, b=13  → Output: -6 (if inputs were reversed)
```

The `%d` format handles negative numbers automatically.

### 4. Data Flow Visualization

```
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  a = 13, b = 7                                           │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (HostToDevice)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   d_a: [13]    d_b: [7]                                  │
│          \      /                                        │
│           \    /                                         │
│            -   ← Subtraction: 13 - 7 = 6                 │
│            │                                             │
│            ▼                                             │
│   d_result: [6]                                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (DeviceToHost)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  result = 6 → printf("6\n")                              │
└──────────────────────────────────────────────────────────┘
```

### 5. Zero Result Case

Example 3 demonstrates when a equals b:

```
a = 50, b = 50
result = 50 - 50 = 0
```

Zero is a valid result and prints correctly with `%d`.

---

## Alternative Solutions

### Using Array

```cuda
__global__ void diffKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        data[2] = data[0] - data[1];  // result at index 2
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
    
    diffKernel<<<1, 1>>>(d_data);
    
    cudaMemcpy(h_data, d_data, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d\n", h_data[2]);
    
    cudaFree(d_data);
    
    return 0;
}
```

### Direct Output in Kernel

```cuda
__global__ void diffKernel(int* a, int* b) {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("%d\n", *a - *b);
    }
}
```

---

## Comparison: Sum vs Difference

| Aspect | Sum of Two Integers | Difference of Two Integers |
|--------|---------------------|---------------------------|
| Operation | `*a + *b` | `*a - *b` |
| Commutative | Yes (a+b = b+a) | No (a-b ≠ b-a) |
| Result range | 2 to 200 | -99 to 99 |
| Zero possible | No (min is 2) | Yes (when a = b) |

---

## Common Mistakes

### ❌ Wrong Operand Order
```cuda
*result = *b - *a;  // Wrong! Should be *a - *b
```
Order matters for subtraction.

### ❌ Forgetting Negative Possibility
```cuda
unsigned int result;  // Can't hold negative values!
int result;  // Correct - can hold negative values
```

### ❌ Swapping Inputs
```cuda
// If problem asks for a - b:
cudaMemcpy(d_a, &b, ...);  // Wrong! Swapped
cudaMemcpy(d_b, &a, ...);  // Wrong! Swapped
```

---

## Arithmetic Operations Reference

| Operation | Symbol | Commutative | Example |
|-----------|--------|-------------|---------|
| Addition | `+` | Yes | 3 + 5 = 5 + 3 = 8 |
| Subtraction | `-` | No | 5 - 3 ≠ 3 - 5 |
| Multiplication | `*` | Yes | 3 × 5 = 5 × 3 = 15 |
| Division | `/` | No | 6 / 2 ≠ 2 / 6 |
| Modulo | `%` | No | 7 % 3 ≠ 3 % 7 |

---

## Key Takeaways

1. **Same pattern as addition** — only operator changes
2. **Order matters** — subtraction is not commutative
3. **Negative results possible** — even with positive inputs
4. **Zero is valid** — when both inputs are equal
5. **`int` handles negatives** — use signed type for result

---

## Practice Exercises

1. Compute **absolute difference** |a - b|
2. Compute **both** sum and difference in one kernel
3. Determine which is larger and print `a > b`, `a < b`, or `a = b`
4. Compute a - b - c for three integers

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/22)*
