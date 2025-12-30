# Equal To

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Given two integers a and b, determine if a == b. Print 1 if true, otherwise print 0.

### Input
The first line contains two integers a and b separated by a space (1 ≤ a, b ≤ 100).

### Output
Print 1 if a == b, otherwise print 0.

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
| 1 7 | 0 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void equalKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*a == *b) ? 1 : 0;
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
    equalKernel<<<1, 1>>>(d_a, d_b, d_result);
    
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
| ✅ Kernel exists | `__global__ void equalKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs equality check on GPU |

---

## CUDA Concepts Covered

### 1. Equality Operator

The `==` operator checks if two values are exactly equal:

```cuda
*result = (*a == *b) ? 1 : 0;

// Examples:
// 5 == 3 → false → 0 (different values)
// 2 == 2 → true  → 1 (same value)
// 1 == 7 → false → 0 (different values)
```

### 2. Equality is Commutative

Unlike `<`, `>`, equality is commutative:

```cuda
a == b  ≡  b == a

// Example: a = 2, b = 2
// 2 == 2 → true
// 2 == 2 → true (same result, order doesn't matter)
```

### 3. Assignment vs Comparison

⚠️ **Critical distinction in C/CUDA:**

```cuda
// COMPARISON (two equal signs)
*result = (*a == *b);  // Check if a equals b

// ASSIGNMENT (one equal sign)
*result = (*a = *b);   // Assign b to a, then assign to result
```

This is one of the most common bugs in C programming!

### 4. Comparison Result Visualization

```
Example 1: a = 5, b = 3
┌───────────────────┐
│   5 == 3 ?        │
│     ↓             │
│   5 ≠ 3           │
│   false → 0       │
└───────────────────┘

Example 2: a = 2, b = 2
┌───────────────────┐
│   2 == 2 ?        │
│     ↓             │
│   2 = 2           │
│   true → 1        │
└───────────────────┘

Example 3: a = 1, b = 7
┌───────────────────┐
│   1 == 7 ?        │
│     ↓             │
│   1 ≠ 7           │
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
│           ==   ← Comparison: 2 == 2 = true               │
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
__global__ void equalKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*a == *b);
    }
}
```

### Using Subtraction Check

```cuda
__global__ void equalKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = ((*a - *b) == 0);  // Equal if difference is zero
    }
}
```

### Using Combined <= and >=

```cuda
__global__ void equalKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // a == b when a <= b AND a >= b
        *result = (*a <= *b) && (*a >= *b);
    }
}
```

---

## Equality's Role in Other Comparisons

`==` is implicitly used in other comparison operators:

| Operator | Definition |
|----------|------------|
| `>=` | `>` OR `==` |
| `<=` | `<` OR `==` |
| `!=` | NOT `==` |

```cuda
a >= b  ≡  (a > b) || (a == b)
a <= b  ≡  (a < b) || (a == b)
a != b  ≡  !(a == b)
```

---

## Common Mistakes

### ❌ Using Single = (Assignment)
```cuda
*result = (*a = *b);   // WRONG! This is assignment
*result = (*a == *b);  // Correct - comparison
```
This is the #1 bug in C/CUDA programming!

### ❌ Comparing Floats with ==
```cuda
float x = 0.1 + 0.2;
float y = 0.3;
if (x == y) { ... }  // May fail due to floating-point precision!

// Better approach for floats:
if (fabs(x - y) < 0.0001) { ... }  // Use epsilon comparison
```

### ❌ Wrong Symbol Order
```cuda
*result = (*a =* *b);  // Wrong! Syntax error
*result = (*a == *b);  // Correct
```

---

## Truth Table for == Operator

| a | b | a == b | a != b |
|---|---|--------|--------|
| 5 | 5 | 1 | 0 |
| 5 | 3 | 0 | 1 |
| 0 | 0 | 1 | 0 |
| 1 | 7 | 0 | 1 |
| 100 | 100 | 1 | 0 |

Notice: `==` and `!=` are **always opposite**.

---

## Use Cases in CUDA

| Use Case | Example | Purpose |
|----------|---------|---------|
| Value matching | `if (arr[i] == target)` | Search |
| Thread selection | `if (threadIdx.x == 0)` | Single-thread execution |
| Sentinel check | `if (value == -1)` | End of data |
| State comparison | `if (state == DONE)` | FSM |
| Divisibility | `if (n % d == 0)` | Check if divisible |

### Example: Finding a Value

```cuda
__global__ void findValue(int* arr, int n, int target, int* found) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        if (arr[idx] == target) {
            *found = 1;  // Found!
        }
    }
}
```

---

## Equality with Different Types

```cuda
// Integer equality - exact match
int a = 5, b = 5;
(a == b) → true

// Character equality - ASCII comparison
char c1 = 'A', c2 = 'A';
(c1 == c2) → true

// Pointer equality - same memory address
int* p1 = &a;
int* p2 = &a;
(p1 == p2) → true

// Float equality - may have precision issues!
float f1 = 0.1f + 0.2f;
float f2 = 0.3f;
(f1 == f2) → may be false!
```

---

## Key Takeaways

1. **`==` checks exact equality** — true only when values are identical
2. **Commutative** — `a == b` equals `b == a`
3. **Use `==`, not `=`** — double equals for comparison, single for assignment
4. **Avoid with floats** — use epsilon comparison instead
5. **Foundation for matching** — essential in search and conditional logic

---

## Practice Exercises

1. Implement **not equal** (a != b)
2. Check if a number is **zero** using ==
3. Find the **index** of a target value in an array
4. Count how many elements **equal** a given value

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/37)*
