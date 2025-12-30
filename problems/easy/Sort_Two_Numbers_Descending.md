# Sort Two Numbers Descending

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Given two integers, output them in descending order.

### Input
Two integers are given in a single line, separated by a space.

### Output
Print the two integers in descending order, separated by a space.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 1 2 | 2 1 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 3 7 | 7 3 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 10 10 | 10 10 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void sortDescendingKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        if (data[0] < data[1]) {
            int temp = data[0];
            data[0] = data[1];
            data[1] = temp;
        }
    }
}

int main() {
    int a, b;
    scanf("%d %d", &a, &b);
    
    // Host array
    int h_data[2] = {a, b};
    
    // Device memory
    int* d_data;
    cudaMalloc(&d_data, 2 * sizeof(int));
    
    // Copy to device
    cudaMemcpy(d_data, h_data, 2 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    sortDescendingKernel<<<1, 1>>>(d_data);
    
    // Copy back to host
    cudaMemcpy(h_data, d_data, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d %d\n", h_data[0], h_data[1]);
    
    cudaFree(d_data);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void sortDescendingKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Conditional swap for descending order |

---

## CUDA Concepts Covered

### 1. Ascending vs Descending: Flip the Comparison

The only difference from ascending sort is the comparison operator:

```cuda
// Ascending order: swap if first > second
if (data[0] > data[1]) { swap(); }  // Result: smaller first

// Descending order: swap if first < second
if (data[0] < data[1]) { swap(); }  // Result: larger first
```

### 2. Comparison Logic Visualization

```
Ascending (a > b → swap):
┌───┬───┐         ┌───┬───┐
│ 7 │ 3 │   →     │ 3 │ 7 │   (smaller, larger)
└───┴───┘         └───┴───┘

Descending (a < b → swap):
┌───┬───┐         ┌───┬───┐
│ 3 │ 7 │   →     │ 7 │ 3 │   (larger, smaller)
└───┴───┘         └───┴───┘
```

### 3. Descending Sort Flow

```
Case 1: Already in descending order (a ≥ b)
┌─────┬─────┐
│  7  │  3  │  →  7 < 3? No  →  No swap
└─────┴─────┘
Output: 7 3

Case 2: Needs swap (a < b)
┌─────┬─────┐
│  3  │  7  │  →  3 < 7? Yes →  Swap!
└─────┴─────┘
       ↓
┌─────┬─────┐
│  7  │  3  │
└─────┴─────┘
Output: 7 3

Case 3: Equal values
┌─────┬─────┐
│ 10  │ 10  │  →  10 < 10? No  →  No swap
└─────┴─────┘
Output: 10 10
```

### 4. Using max/min for Descending Order

An alternative approach using helper functions:

```cuda
__device__ int d_max(int a, int b) { return (a > b) ? a : b; }
__device__ int d_min(int a, int b) { return (a < b) ? a : b; }

__global__ void sortDescendingKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int a = data[0];
        int b = data[1];
        data[0] = d_max(a, b);  // Larger value first
        data[1] = d_min(a, b);  // Smaller value second
    }
}
```

### 5. Comparison Table: Ascending vs Descending

| Aspect | Ascending | Descending |
|--------|-----------|------------|
| Swap condition | `a > b` | `a < b` |
| First element | Smaller | Larger |
| Second element | Larger | Smaller |
| Result | `[min, max]` | `[max, min]` |

---

## Alternative Solutions

### Using Ternary Operators

```cuda
__global__ void sortDescendingKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int a = data[0];
        int b = data[1];
        data[0] = (a > b) ? a : b;  // max
        data[1] = (a < b) ? a : b;  // min
    }
}
```

### Using Arithmetic (Branchless)

```cuda
__global__ void sortDescendingBranchless(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int a = data[0];
        int b = data[1];
        int diff = a - b;
        int sign = (diff >> 31) & 1;  // 1 if a < b, 0 otherwise
        
        // Swap using sign as selector
        data[0] = a + sign * (b - a);  // b if a < b, else a
        data[1] = b - sign * (b - a);  // a if a < b, else b
    }
}
```

> ⚠️ **Note**: Branchless versions can be useful for GPU performance but may be harder to read. Use them when optimizing parallel sorting algorithms.

---

## Common Mistakes

### ❌ Using Wrong Comparison Operator
```cuda
// This sorts in ASCENDING order, not descending!
if (data[0] > data[1]) {
    // swap
}
```
For descending order, use `<` not `>`.

### ❌ Confusing Output Order
```cuda
// Wrong: outputting in wrong order after correct sort
printf("%d %d\n", h_data[1], h_data[0]);  // Reversed output!
```

### ❌ Swapping When Equal
```cuda
if (data[0] <= data[1]) {  // Wrong! This swaps equal values unnecessarily
    // swap
}
```
Use strict `<` comparison; equal values don't need swapping.

### ❌ Copy-Paste Error from Ascending Sort
```cuda
// Copied from ascending sort but forgot to change condition
if (data[0] > data[1]) {  // Should be < for descending!
    int temp = data[0];
    data[0] = data[1];
    data[1] = temp;
}
```

---

## Relationship to Previous Problem

This problem is the complement of [Sort Two Numbers (Ascending)](Sort_Two_Numbers.md):

| Problem | Swap Condition | Output Format |
|---------|----------------|---------------|
| Sort Ascending | `a > b` | `[smaller, larger]` |
| Sort Descending | `a < b` | `[larger, smaller]` |

Understanding both helps build intuition for comparison-based sorting algorithms.

---

## Key Takeaways

1. **Flip the comparison** operator to change sort direction
2. **Descending order** means larger values come first
3. **Same swap logic** — only the condition changes
4. **Equal values** require no action in either direction
5. **Foundation for sorting** — this pattern scales to larger arrays

---

## Practice Exercises

1. Create a kernel that takes a `direction` parameter (0=ascending, 1=descending)
2. Sort three numbers in descending order
3. Verify behavior with negative numbers (e.g., `-5 3`)
4. Compare performance of conditional swap vs branchless versions

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/11)*
