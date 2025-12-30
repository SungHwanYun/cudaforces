# Sort Two Numbers

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Given two integers, output them in ascending order.

### Input
Two integers are given in a single line, separated by a space.

### Output
Print the two integers in ascending order, separated by a space.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 2 1 | 1 2 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 7 3 | 3 7 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 10 10 | 10 10 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void sortTwoKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        if (data[0] > data[1]) {
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
    sortTwoKernel<<<1, 1>>>(d_data);
    
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
| ✅ Kernel exists | `__global__ void sortTwoKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Conditional swap to sort values |

---

## CUDA Concepts Covered

### 1. Conditional Logic in Kernels

Kernels can contain any standard C control flow:

```cuda
__global__ void sortTwoKernel(int* data) {
    if (idx == 0) {
        if (data[0] > data[1]) {  // Conditional comparison
            // Swap only if needed
            int temp = data[0];
            data[0] = data[1];
            data[1] = temp;
        }
    }
}
```

### 2. Comparison-Based Sorting

The simplest sort for two elements: compare and conditionally swap:

```cuda
if (data[0] > data[1]) {
    // data[0] is larger, so swap to put smaller first
    swap(data[0], data[1]);
}
// After this: data[0] ≤ data[1] (ascending order)
```

### 3. Handling Equal Values

When both values are equal, no swap is needed:

```
Input: 10 10

data[0] = 10, data[1] = 10
Condition: 10 > 10 → false
No swap performed
Output: 10 10 ✓
```

### 4. Sorting Flow Visualization

```
Case 1: Already sorted (a ≤ b)
┌─────┬─────┐
│  3  │  7  │  →  3 > 7? No  →  No swap
└─────┴─────┘
Output: 3 7

Case 2: Needs swap (a > b)
┌─────┬─────┐
│  7  │  3  │  →  7 > 3? Yes →  Swap!
└─────┴─────┘
       ↓
┌─────┬─────┐
│  3  │  7  │
└─────┴─────┘
Output: 3 7

Case 3: Equal values
┌─────┬─────┐
│ 10  │ 10  │  →  10 > 10? No  →  No swap
└─────┴─────┘
Output: 10 10
```

### 5. Branchless Alternative

For performance-critical code, you can avoid branches:

```cuda
__global__ void sortTwoBranchless(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int a = data[0];
        int b = data[1];
        int min_val = (a < b) ? a : b;  // or: a * (a < b) + b * (a >= b)
        int max_val = (a > b) ? a : b;
        data[0] = min_val;
        data[1] = max_val;
    }
}
```

---

## Alternative Solutions

### Using min/max Functions

```cuda
__device__ int d_min(int a, int b) {
    return (a < b) ? a : b;
}

__device__ int d_max(int a, int b) {
    return (a > b) ? a : b;
}

__global__ void sortTwoKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int a = data[0];
        int b = data[1];
        data[0] = d_min(a, b);
        data[1] = d_max(a, b);
    }
}
```

### XOR Swap (No Temporary Variable)

```cuda
__global__ void sortTwoXorKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        if (data[0] > data[1]) {
            data[0] ^= data[1];
            data[1] ^= data[0];
            data[0] ^= data[1];
        }
    }
}
```

> ⚠️ **Note**: XOR swap fails when swapping a variable with itself. Safe here since we have two distinct array elements.

---

## Common Mistakes

### ❌ Wrong Comparison Direction
```cuda
if (data[0] < data[1]) {  // Wrong! This swaps when already sorted
    // swap
}
```
For ascending order, swap when `data[0] > data[1]`.

### ❌ Forgetting the Equal Case
```cuda
// This is actually fine - equal values don't need swapping
// But be aware that your logic should handle it correctly
if (data[0] >= data[1]) {  // This would swap equal values unnecessarily
    // swap
}
```

### ❌ Swapping Without Temporary Variable (Incorrectly)
```cuda
if (data[0] > data[1]) {
    data[0] = data[1];  // Oops! Original data[0] is lost
    data[1] = data[0];  // Now both have the same value
}
```

### ❌ Comparing Before Copying to Device
```cuda
int main() {
    int a, b;
    scanf("%d %d", &a, &b);
    
    // Wrong: sorting on host, not using GPU
    if (a > b) {
        int temp = a; a = b; b = temp;
    }
    printf("%d %d\n", a, b);  // No CUDA operations!
}
```

---

## Sorting Fundamentals

This problem introduces the most basic sorting concept:

| Elements | Comparisons Needed | Algorithm |
|----------|-------------------|-----------|
| 2 | 1 | Compare & Swap |
| 3 | 3 | Insertion/Selection |
| N | O(N log N) | Merge/Quick Sort |
| N (parallel) | O(log N) | Bitonic Sort |

For two elements, a single comparison is optimal.

---

## Key Takeaways

1. **Conditional swap** is the foundation of comparison-based sorting
2. **No action needed** when elements are already in order or equal
3. **Temporary variable** prevents data loss during swap
4. **Branchless alternatives** can improve GPU performance
5. **Simple problems** build intuition for parallel sorting algorithms

---

## Practice Exercises

1. Modify to sort in **descending** order
2. Extend to sort **three** numbers
3. Handle negative numbers (verify the solution works correctly)
4. Implement parallel comparison for an array of pairs

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/10)*
