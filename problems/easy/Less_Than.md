# Less Than

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Given two integers a and b, determine if a < b. Print 1 if true, otherwise print 0.

### Input
The first line contains two integers a and b separated by a space (1 ≤ a, b ≤ 100).

### Output
Print 1 if a < b, otherwise print 0.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 5 3 | 0 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 2 2 | 0 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 1 7 | 1 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void lessThanKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*a < *b) ? 1 : 0;
    }
}

int main() {
    int a, b;
    scanf("%d %d", &a, &b);
    
    // Device memory
    int *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    // Copy inputs to device
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    lessThanKernel<<<1, 1>>>(d_a, d_b, d_result);
    
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
| ✅ Kernel exists | `__global__ void lessThanKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs comparison on GPU |

---

## CUDA Concepts Covered

### 1. Less Than Operator

The `<` operator checks if the first operand is strictly less than the second:

```cuda
*result = (*a < *b) ? 1 : 0;

// Examples:
// 5 < 3 → false → 0 (5 is greater than 3)
// 2 < 2 → false → 0 (2 equals 2, not less than)
// 1 < 7 → true  → 1 (1 is less than 7)
```

### 2. Relationship with Greater Than

`<` and `>` are mirror operators:

```cuda
a < b  ≡  b > a

// Example: a = 1, b = 7
// 1 < 7 → true
// 7 > 1 → true (same result, operands swapped)
```

### 3. Comparison Result Visualization

```
Example 1: a = 5, b = 3
┌───────────────────┐
│   5 < 3 ?         │
│     ↓             │
│   5 is greater    │
│   false → 0       │
└───────────────────┘

Example 2: a = 2, b = 2
┌───────────────────┐
│   2 < 2 ?         │
│     ↓             │
│   2 equals 2      │
│   false → 0       │  (equal is NOT less than)
└───────────────────┘

Example 3: a = 1, b = 7
┌───────────────────┐
│   1 < 7 ?         │
│     ↓             │
│   1 is smaller    │
│   true → 1        │
└───────────────────┘
```

### 4. Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  a = 1, b = 7                                            │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (HostToDevice)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   d_a: [1]     d_b: [7]                                  │
│          \      /                                        │
│           \    /                                         │
│            <   ← Comparison: 1 < 7 = true                │
│            │                                             │
│            ▼                                             │
│   d_result: [1]                                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (DeviceToHost)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  result = 1 → printf("1\n")                              │
└──────────────────────────────────────────────────────────┘
```

### 5. Bounds Checking Pattern

`<` is the most common operator for array bounds checking in CUDA:

```cuda
__global__ void processArray(int* arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {  // Standard bounds check
        arr[idx] = arr[idx] * 2;
    }
}
```

---

## Alternative Solutions

### Using Implicit Boolean Conversion

```cuda
__global__ void lessThanKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*a < *b);
    }
}
```

### Using Negation of >=

```cuda
__global__ void lessThanKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = !(*a >= *b);  // NOT (a >= b) equals (a < b)
    }
}
```

### Swapping Operands with >

```cuda
__global__ void lessThanKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*b > *a);  // b > a equals a < b
    }
}
```

---

## Comparison: > vs <

| a | b | a > b | a < b |
|---|---|-------|-------|
| 5 | 3 | 1 | 0 |
| 3 | 5 | 0 | 1 |
| 5 | 5 | 0 | 0 |
| 1 | 7 | 0 | 1 |
| 7 | 1 | 1 | 0 |

Notice: `a > b` and `a < b` are **opposites when a ≠ b**, but **both false when a = b**.

---

## Common Mistakes

### ❌ Confusing < with >
```cuda
*result = (*a > *b);  // Wrong! This is "greater than"
*result = (*a < *b);  // Correct - "less than"
```

### ❌ Including Equal Case
```cuda
// < does NOT include equal
// 2 < 2 returns 0 (false)
// Use <= if you want "less than or equal"
```

### ❌ Reversing Operands
```cuda
*result = (*b < *a);  // Wrong! This checks if b < a
*result = (*a < *b);  // Correct - checks if a < b
```

---

## Operator Relationships

```
a < b   ≡   b > a       (swap operands)
a < b   ≡   !(a >= b)   (negate opposite)
a <= b  ≡   !(a > b)    (negate strict)
```

These equivalences are useful for understanding and optimization.

---

## Why < is Most Common in CUDA

The `<` operator is the standard for bounds checking because:

```cuda
// Array of n elements: valid indices are 0 to n-1
// Check: idx < n

n = 5:
  idx = 0: 0 < 5 ✓ valid
  idx = 1: 1 < 5 ✓ valid
  idx = 2: 2 < 5 ✓ valid
  idx = 3: 3 < 5 ✓ valid
  idx = 4: 4 < 5 ✓ valid
  idx = 5: 5 < 5 ✗ invalid (out of bounds)
```

---

## Truth Table for < Operator

| a | b | a < b | Description |
|---|---|-------|-------------|
| 1 | 7 | 1 | a is smaller |
| 7 | 1 | 0 | a is larger |
| 5 | 5 | 0 | equal (not less) |
| 0 | 1 | 1 | a is smaller |
| 100 | 99 | 0 | a is larger |

---

## Key Takeaways

1. **`<` is strict** — does not include equality
2. **Mirror of `>`** — `a < b` equals `b > a`
3. **Most common in bounds checking** — `if (idx < n)`
4. **False when equal** — `5 < 5` is false
5. **Essential for safe array access** in parallel programming

---

## Practice Exercises

1. Implement **less than or equal** (a <= b)
2. Find the **minimum** of two numbers using <
3. Count how many elements are **less than** a threshold
4. Implement **sorting** using < comparisons

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/35)*
