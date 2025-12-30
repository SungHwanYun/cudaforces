# Greater Than or Equal To

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Given two integers a and b, determine if a ≥ b. Print 1 if true, otherwise print 0.

### Input
The first line contains two integers a and b separated by a space (1 ≤ a, b ≤ 100).

### Output
Print 1 if a ≥ b, otherwise print 0.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 5 3 | 1 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 2 2 | 1 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 1 7 | 0 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void greaterEqualKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*a >= *b) ? 1 : 0;
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
    greaterEqualKernel<<<1, 1>>>(d_a, d_b, d_result);
    
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
| ✅ Kernel exists | `__global__ void greaterEqualKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs comparison on GPU |

---

## CUDA Concepts Covered

### 1. Greater Than or Equal Operator

The `>=` operator combines greater than and equality:

```cuda
*result = (*a >= *b) ? 1 : 0;

// Examples:
// 5 >= 3 → true  → 1 (5 is greater than 3)
// 2 >= 2 → true  → 1 (2 equals 2)
// 1 >= 7 → false → 0 (1 is less than 7)
```

### 2. Difference from Greater Than

```
a > b:  Returns true only if a is strictly greater than b
a >= b: Returns true if a is greater than OR equal to b

Example with a = 2, b = 2:
  2 > 2  → false (0)
  2 >= 2 → true (1)  ← This is the difference!
```

### 3. Equivalent Expressions

`a >= b` can be expressed in multiple ways:

```cuda
// Direct
*result = (*a >= *b);

// Using NOT and <
*result = !(*a < *b);

// Using OR
*result = (*a > *b) || (*a == *b);
```

### 4. Comparison Result Visualization

```
Example 1: a = 5, b = 3
┌───────────────────┐
│   5 >= 3 ?        │
│     ↓             │
│   5 > 3 ? Yes!    │
│   true → 1        │
└───────────────────┘

Example 2: a = 2, b = 2
┌───────────────────┐
│   2 >= 2 ?        │
│     ↓             │
│   2 > 2 ? No      │
│   2 == 2 ? Yes!   │
│   true → 1        │
└───────────────────┘

Example 3: a = 1, b = 7
┌───────────────────┐
│   1 >= 7 ?        │
│     ↓             │
│   1 > 7 ? No      │
│   1 == 7 ? No     │
│   false → 0       │
└───────────────────┘
```

### 5. Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  a = 2, b = 2                                            │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (HostToDevice)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   d_a: [2]     d_b: [2]                                  │
│          \      /                                        │
│           \    /                                         │
│           >=   ← Comparison: 2 >= 2 = true               │
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

---

## Alternative Solutions

### Using Implicit Boolean Conversion

```cuda
__global__ void greaterEqualKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*a >= *b);
    }
}
```

### Using Negation of Less Than

```cuda
__global__ void greaterEqualKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = !(*a < *b);  // NOT (a < b) equals (a >= b)
    }
}
```

### Using OR Logic

```cuda
__global__ void greaterEqualKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*a > *b) || (*a == *b);
    }
}
```

---

## Comparison: > vs >=

| a | b | a > b | a >= b |
|---|---|-------|--------|
| 5 | 3 | 1 | 1 |
| 3 | 5 | 0 | 0 |
| 5 | 5 | **0** | **1** |
| 0 | 0 | **0** | **1** |
| 100 | 100 | **0** | **1** |

The key difference is when **a equals b**.

---

## Common Mistakes

### ❌ Using > Instead of >=
```cuda
*result = (*a > *b);   // Wrong! Returns 0 when a == b
*result = (*a >= *b);  // Correct - returns 1 when a == b
```

### ❌ Wrong Operator Symbol
```cuda
*result = (*a => *b);  // Wrong! Not a valid operator
*result = (*a >= *b);  // Correct syntax
```

### ❌ Confusing >= with > and ==
```cuda
// >= is a single operator, not two checks
*result = (*a > *b == *b);  // Wrong! This is nonsense
*result = (*a >= *b);       // Correct
```

---

## Comparison Operators Summary

| Operator | Name | Includes Equal? | Example (a=5) |
|----------|------|-----------------|---------------|
| `>` | Greater than | No | `5 > 5` → 0 |
| `>=` | Greater or equal | **Yes** | `5 >= 5` → 1 |
| `<` | Less than | No | `5 < 5` → 0 |
| `<=` | Less or equal | **Yes** | `5 <= 5` → 1 |
| `==` | Equal | (is equality) | `5 == 5` → 1 |
| `!=` | Not equal | (is inequality) | `5 != 5` → 0 |

---

## Use Cases in CUDA

| Use Case | Operator | Example |
|----------|----------|---------|
| Bounds check (inclusive) | `>=` | `if (idx >= 0)` |
| Array end check (inclusive) | `<=` | `if (idx <= n-1)` |
| Minimum threshold | `>=` | `if (value >= minValue)` |
| Range check | `>=` and `<=` | `if (x >= lo && x <= hi)` |

### Example: Inclusive Range Check

```cuda
__global__ void processInRange(int* arr, int n, int lo, int hi) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        // Process only values in range [lo, hi]
        if (arr[idx] >= lo && arr[idx] <= hi) {
            arr[idx] = arr[idx] * 2;
        }
    }
}
```

---

## Logical Equivalences

```
a >= b  ≡  !(a < b)      (NOT less than)
a >= b  ≡  a > b || a == b   (greater OR equal)
a >= b  ≡  b <= a        (b is less than or equal to a)
```

These equivalences can be useful for code optimization or readability.

---

## Key Takeaways

1. **`>=` includes equality** — returns true when a equals b
2. **Key difference from `>`** — `5 > 5` is false, `5 >= 5` is true
3. **Common in bounds checking** — inclusive range validation
4. **Logical equivalence** — `!(a < b)` equals `a >= b`
5. **Foundation for range checks** — often paired with `<=`

---

## Practice Exercises

1. Implement **less than or equal** (a <= b)
2. Check if a value is in **inclusive range** [lo, hi]
3. Find the **minimum** of two numbers using >=
4. Implement a **clamp function**: ensure value is between min and max

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/34)*
