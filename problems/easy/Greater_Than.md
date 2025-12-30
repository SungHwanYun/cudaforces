# Greater Than

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Given two integers a and b, determine if a > b. Print 1 if true, otherwise print 0.

### Input
The first line contains two integers a and b separated by a space (1 ≤ a, b ≤ 100).

### Output
Print 1 if a > b, otherwise print 0.

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
| 1 7 | 0 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void greaterThanKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*a > *b) ? 1 : 0;
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
    greaterThanKernel<<<1, 1>>>(d_a, d_b, d_result);
    
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
| ✅ Kernel exists | `__global__ void greaterThanKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs comparison on GPU |

---

## CUDA Concepts Covered

### 1. Comparison Operators

The `>` operator returns a boolean result (true/false):

```cuda
*result = (*a > *b) ? 1 : 0;

// Examples:
// 5 > 3 → true  → 1
// 2 > 2 → false → 0
// 1 > 7 → false → 0
```

### 2. Ternary Operator

The `? :` operator provides a compact if-else:

```cuda
// Ternary form:
*result = (*a > *b) ? 1 : 0;

// Equivalent if-else:
if (*a > *b) {
    *result = 1;
} else {
    *result = 0;
}
```

### 3. Boolean to Integer Conversion

In C/CUDA, comparison results are already integers:

```cuda
// Implicit conversion (also valid):
*result = (*a > *b);  // true → 1, false → 0

// Explicit ternary (clearer):
*result = (*a > *b) ? 1 : 0;
```

### 4. Comparison Result Visualization

```
Example 1: a = 5, b = 3
┌───────────────────┐
│   5 > 3 ?         │
│     ↓             │
│   true → 1        │
└───────────────────┘

Example 2: a = 2, b = 2
┌───────────────────┐
│   2 > 2 ?         │
│     ↓             │
│   false → 0       │  (equal is NOT greater)
└───────────────────┘

Example 3: a = 1, b = 7
┌───────────────────┐
│   1 > 7 ?         │
│     ↓             │
│   false → 0       │
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
│            >   ← Comparison: 5 > 3 = true                │
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
__global__ void greaterThanKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*a > *b);  // Implicit: true→1, false→0
    }
}
```

### Using If-Else

```cuda
__global__ void greaterThanKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        if (*a > *b) {
            *result = 1;
        } else {
            *result = 0;
        }
    }
}
```

### Using Array

```cuda
__global__ void greaterThanKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        data[2] = (data[0] > data[1]) ? 1 : 0;
    }
}
```

---

## Comparison Operators Reference

| Operator | Meaning | Example | Result |
|----------|---------|---------|--------|
| `>` | Greater than | `5 > 3` | 1 (true) |
| `<` | Less than | `5 < 3` | 0 (false) |
| `>=` | Greater or equal | `5 >= 5` | 1 (true) |
| `<=` | Less or equal | `5 <= 3` | 0 (false) |
| `==` | Equal | `5 == 5` | 1 (true) |
| `!=` | Not equal | `5 != 3` | 1 (true) |

---

## Common Mistakes

### ❌ Using Assignment Instead of Comparison
```cuda
*result = (*a = *b);  // Wrong! This is assignment (=)
*result = (*a > *b);  // Correct - comparison (>)
```

### ❌ Forgetting That Equal is Not Greater
```cuda
// When a == b:
*result = (*a > *b);  // Returns 0, not 1
// Use >= if you want "greater than or equal"
```

### ❌ Wrong Output Format
```cuda
printf("true\n");   // Wrong! Should print 1 or 0
printf("%d\n", result);  // Correct
```

### ❌ Reversing Operands
```cuda
*result = (*b > *a);  // Wrong! This checks if b > a
*result = (*a > *b);  // Correct - checks if a > b
```

---

## Comparison Use Cases in CUDA

| Use Case | Example | Purpose |
|----------|---------|---------|
| Bounds checking | `idx < n` | Prevent out-of-bounds access |
| Conditional execution | `if (value > threshold)` | Branching |
| Finding maximum | `a > b ? a : b` | Max selection |
| Sorting | `if (arr[i] > arr[j]) swap()` | Compare-swap |
| Early termination | `if (error > tolerance)` | Algorithm control |

### Example: Bounds Checking in Parallel

```cuda
__global__ void processArray(int* arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {  // Bounds check using comparison
        arr[idx] = arr[idx] * 2;
    }
}
```

---

## Truth Table for > Operator

| a | b | a > b |
|---|---|-------|
| 5 | 3 | 1 |
| 3 | 5 | 0 |
| 5 | 5 | 0 |
| 0 | 0 | 0 |
| 100 | 1 | 1 |
| 1 | 100 | 0 |

---

## Key Takeaways

1. **Comparison returns boolean** — true (1) or false (0)
2. **Equal is not greater** — `a > a` is always false
3. **Ternary operator** provides concise conditional assignment
4. **Foundation for control flow** — essential in bounds checking and branching
5. **Building block for sorting** — comparisons drive sorting algorithms

---

## Practice Exercises

1. Implement **less than** comparison (a < b)
2. Implement **greater than or equal** (a >= b)
3. Find the **maximum** of two numbers using comparison
4. Compare and output "YES" or "NO" instead of 1 or 0

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/33)*
