# Less Than or Equal To

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Given two integers a and b, determine if a ≤ b. Print 1 if true, otherwise print 0.

### Input
The first line contains two integers a and b separated by a space (1 ≤ a, b ≤ 100).

### Output
Print 1 if a ≤ b, otherwise print 0.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 5 3 | 0 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 2 2 | 1 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 1 7 | 1 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void lessEqualKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*a <= *b) ? 1 : 0;
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
    lessEqualKernel<<<1, 1>>>(d_a, d_b, d_result);
    
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
| ✅ Kernel exists | `__global__ void lessEqualKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs comparison on GPU |

---

## CUDA Concepts Covered

### 1. Less Than or Equal Operator

The `<=` operator combines less than and equality:

```cuda
*result = (*a <= *b) ? 1 : 0;

// Examples:
// 5 <= 3 → false → 0 (5 is greater than 3)
// 2 <= 2 → true  → 1 (2 equals 2)
// 1 <= 7 → true  → 1 (1 is less than 7)
```

### 2. Difference from Less Than

```
a < b:  Returns true only if a is strictly less than b
a <= b: Returns true if a is less than OR equal to b

Example with a = 2, b = 2:
  2 < 2  → false (0)
  2 <= 2 → true (1)  ← This is the difference!
```

### 3. Comparison Result Visualization

```
Example 1: a = 5, b = 3
┌───────────────────┐
│   5 <= 3 ?        │
│     ↓             │
│   5 < 3 ? No      │
│   5 == 3 ? No     │
│   false → 0       │
└───────────────────┘

Example 2: a = 2, b = 2
┌───────────────────┐
│   2 <= 2 ?        │
│     ↓             │
│   2 < 2 ? No      │
│   2 == 2 ? Yes!   │
│   true → 1        │
└───────────────────┘

Example 3: a = 1, b = 7
┌───────────────────┐
│   1 <= 7 ?        │
│     ↓             │
│   1 < 7 ? Yes!    │
│   true → 1        │
└───────────────────┘
```

### 4. Data Flow

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
│           <=   ← Comparison: 2 <= 2 = true               │
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

### 5. Logical Equivalences

`a <= b` can be expressed in multiple ways:

```cuda
// Direct
*result = (*a <= *b);

// Using NOT and >
*result = !(*a > *b);

// Using OR
*result = (*a < *b) || (*a == *b);

// Mirror relationship
*result = (*b >= *a);
```

---

## Alternative Solutions

### Using Implicit Boolean Conversion

```cuda
__global__ void lessEqualKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*a <= *b);
    }
}
```

### Using Negation of >

```cuda
__global__ void lessEqualKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = !(*a > *b);  // NOT (a > b) equals (a <= b)
    }
}
```

### Using OR Logic

```cuda
__global__ void lessEqualKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*a < *b) || (*a == *b);
    }
}
```

---

## Comparison: < vs <=

| a | b | a < b | a <= b |
|---|---|-------|--------|
| 5 | 3 | 0 | 0 |
| 3 | 5 | 1 | 1 |
| 5 | 5 | **0** | **1** |
| 2 | 2 | **0** | **1** |
| 0 | 0 | **0** | **1** |

The key difference is when **a equals b**.

---

## Complete Comparison Operators Summary

| Operator | Name | Symbol | Includes Equal? |
|----------|------|--------|-----------------|
| `>` | Greater than | > | No |
| `>=` | Greater or equal | ≥ | Yes |
| `<` | Less than | < | No |
| `<=` | Less or equal | ≤ | Yes |
| `==` | Equal | = | (is equality) |
| `!=` | Not equal | ≠ | (is inequality) |

---

## Common Mistakes

### ❌ Using < Instead of <=
```cuda
*result = (*a < *b);   // Wrong! Returns 0 when a == b
*result = (*a <= *b);  // Correct - returns 1 when a == b
```

### ❌ Wrong Operator Symbol
```cuda
*result = (*a =< *b);  // Wrong! Not a valid operator
*result = (*a <= *b);  // Correct syntax (less-than first, then equals)
```

### ❌ Confusing <= with =>
```cuda
*result = (*a => *b);  // Wrong! This is not a comparison operator
*result = (*a <= *b);  // Correct
```

---

## Use Cases in CUDA

| Use Case | Operator | Example |
|----------|----------|---------|
| Last valid index | `<=` | `if (idx <= n-1)` |
| Upper bound check | `<=` | `if (value <= maxValue)` |
| Range check | `>=` and `<=` | `if (x >= lo && x <= hi)` |
| Inclusive iteration | `<=` | `for (i = 0; i <= n; i++)` |

### Example: Inclusive Range Check

```cuda
__global__ void clampValues(int* arr, int n, int minVal, int maxVal) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        // Clamp value to [minVal, maxVal]
        if (arr[idx] <= minVal) {
            arr[idx] = minVal;
        } else if (arr[idx] >= maxVal) {
            arr[idx] = maxVal;
        }
    }
}
```

---

## Operator Pair Relationships

```
Strict vs Inclusive:
  >  and >=  (greater family)
  <  and <=  (less family)

Mirror relationships:
  a < b   ≡   b > a
  a <= b  ≡   b >= a

Negation relationships:
  !(a > b)  ≡  a <= b
  !(a < b)  ≡  a >= b
  !(a >= b) ≡  a < b
  !(a <= b) ≡  a > b
```

---

## Key Takeaways

1. **`<=` includes equality** — returns true when a equals b
2. **Key difference from `<`** — `5 < 5` is false, `5 <= 5` is true
3. **Useful for upper bounds** — inclusive range validation
4. **Logical equivalence** — `!(a > b)` equals `a <= b`
5. **Completes the comparison set** — paired with `>=` for inclusive checks

---

## Practice Exercises

1. Check if a value is in **inclusive range** [lo, hi] using >= and <=
2. Implement a **clamp function** that bounds a value between min and max
3. Count elements that are **at most** a given threshold
4. Compare all six operators in a single program

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/36)*
