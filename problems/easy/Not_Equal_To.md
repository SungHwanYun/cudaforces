# Not Equal To

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Given two integers a and b, determine if a ≠ b. Print 1 if true, otherwise print 0.

### Input
The first line contains two integers a and b separated by a space (1 ≤ a, b ≤ 100).

### Output
Print 1 if a ≠ b, otherwise print 0.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 5 3 | 1 |

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
__global__ void notEqualKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*a != *b) ? 1 : 0;
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
    notEqualKernel<<<1, 1>>>(d_a, d_b, d_result);
    
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
| ✅ Kernel exists | `__global__ void notEqualKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs inequality check on GPU |

---

## CUDA Concepts Covered

### 1. Inequality Operator

The `!=` operator checks if two values are different:

```cuda
*result = (*a != *b) ? 1 : 0;

// Examples:
// 5 != 3 → true  → 1 (different values)
// 2 != 2 → false → 0 (same value)
// 1 != 7 → true  → 1 (different values)
```

### 2. Relationship with Equality

`!=` is the logical negation of `==`:

```cuda
a != b  ≡  !(a == b)

// They are always opposite:
// a == b returns 1 → a != b returns 0
// a == b returns 0 → a != b returns 1
```

### 3. Inequality is Commutative

Like equality, inequality is commutative:

```cuda
a != b  ≡  b != a

// Example: a = 5, b = 3
// 5 != 3 → true
// 3 != 5 → true (same result)
```

### 4. Comparison Result Visualization

```
Example 1: a = 5, b = 3
┌───────────────────┐
│   5 != 3 ?        │
│     ↓             │
│   5 ≠ 3           │
│   true → 1        │
└───────────────────┘

Example 2: a = 2, b = 2
┌───────────────────┐
│   2 != 2 ?        │
│     ↓             │
│   2 = 2           │
│   false → 0       │
└───────────────────┘

Example 3: a = 1, b = 7
┌───────────────────┐
│   1 != 7 ?        │
│     ↓             │
│   1 ≠ 7           │
│   true → 1        │
└───────────────────┘
```

### 5. Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  a = 5, b = 3                                            │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (HostToDevice)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   d_a: [5]     d_b: [3]                                  │
│          \      /                                        │
│           \    /                                         │
│           !=   ← Comparison: 5 != 3 = true               │
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
__global__ void notEqualKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*a != *b);
    }
}
```

### Using Negation of ==

```cuda
__global__ void notEqualKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = !(*a == *b);  // NOT equal
    }
}
```

### Using XOR for Integers

```cuda
__global__ void notEqualKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // XOR is non-zero when values differ
        *result = ((*a ^ *b) != 0);
    }
}
```

---

## Comparison: == vs !=

| a | b | a == b | a != b |
|---|---|--------|--------|
| 5 | 5 | 1 | 0 |
| 5 | 3 | 0 | 1 |
| 0 | 0 | 1 | 0 |
| 1 | 7 | 0 | 1 |
| 2 | 2 | 1 | 0 |

`==` and `!=` are **always opposite** (their sum is always 1).

---

## Common Mistakes

### ❌ Using =! Instead of !=
```cuda
*result = (*a =! *b);  // Wrong! This is assignment + NOT
*result = (*a != *b);  // Correct - not equal comparison
```

### ❌ Double Negation Confusion
```cuda
*result = !(*a != *b);  // This equals (*a == *b)
*result = (*a != *b);   // Just use != directly
```

### ❌ Using <> Symbol
```cuda
*result = (*a <> *b);  // Wrong! Not valid in C/CUDA
*result = (*a != *b);  // Correct C syntax
```

---

## Complete Comparison Operators Summary

| Operator | Symbol | Meaning | Example |
|----------|--------|---------|---------|
| `>` | > | Greater than | `5 > 3` → 1 |
| `>=` | ≥ | Greater or equal | `5 >= 5` → 1 |
| `<` | < | Less than | `3 < 5` → 1 |
| `<=` | ≤ | Less or equal | `5 <= 5` → 1 |
| `==` | = | Equal | `5 == 5` → 1 |
| `!=` | ≠ | Not equal | `5 != 3` → 1 |

---

## Use Cases in CUDA

| Use Case | Example | Purpose |
|----------|---------|---------|
| Exclude threads | `if (threadIdx.x != 0)` | Skip thread 0 |
| Filter values | `if (arr[i] != 0)` | Non-zero elements |
| Error checking | `if (status != SUCCESS)` | Error handling |
| Change detection | `if (oldVal != newVal)` | Update tracking |
| Sparse data | `if (element != 0)` | Skip empty entries |

### Example: Processing Non-Zero Elements

```cuda
__global__ void processNonZero(int* arr, int* result, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        if (arr[idx] != 0) {  // Only process non-zero
            result[idx] = arr[idx] * 2;
        } else {
            result[idx] = 0;
        }
    }
}
```

### Example: Change Detection

```cuda
__global__ void detectChanges(int* old, int* new, int* changed, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        changed[idx] = (old[idx] != new[idx]) ? 1 : 0;
    }
}
```

---

## Logical Relationships

```
Complementary pairs:
  ==  and  !=  (always opposite)
  <   and  >=  (always opposite)
  >   and  <=  (always opposite)

Equivalences:
  a != b  ≡  !(a == b)
  a != b  ≡  (a < b) || (a > b)
  a != b  ≡  !(a <= b && a >= b)
```

---

## Key Takeaways

1. **`!=` checks inequality** — true when values are different
2. **Opposite of `==`** — `!(a == b)` equals `a != b`
3. **Commutative** — `a != b` equals `b != a`
4. **Use `!=`, not `<>`** — correct C/CUDA syntax
5. **Common in filtering** — exclude specific values or threads

---

## Practice Exercises

1. Count elements that are **not equal** to a target
2. Find positions where two arrays **differ**
3. Implement a **change flag** for modified values
4. Filter out **zero elements** from an array

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/38)*
