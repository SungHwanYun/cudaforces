# Sort Three Numbers

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Given three integers, output them in **ascending order**.

### Input
Three integers are given in a single line, separated by spaces.

### Output
Print the three integers in ascending order, separated by spaces.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 3 1 2 | 1 2 3 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 7 7 3 | 3 7 7 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 1 2 3 | 1 2 3 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void sortThreeKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // Bubble sort style: 3 comparisons
        if (data[0] > data[1]) {
            int temp = data[0];
            data[0] = data[1];
            data[1] = temp;
        }
        if (data[1] > data[2]) {
            int temp = data[1];
            data[1] = data[2];
            data[2] = temp;
        }
        if (data[0] > data[1]) {
            int temp = data[0];
            data[0] = data[1];
            data[1] = temp;
        }
    }
}

int main() {
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);
    
    // Host array
    int h_data[3] = {a, b, c};
    
    // Device memory
    int* d_data;
    cudaMalloc(&d_data, 3 * sizeof(int));
    
    // Copy to device
    cudaMemcpy(d_data, h_data, 3 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    sortThreeKernel<<<1, 1>>>(d_data);
    
    // Copy back to host
    cudaMemcpy(h_data, d_data, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d %d %d\n", h_data[0], h_data[1], h_data[2]);
    
    cudaFree(d_data);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void sortThreeKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Three comparison-swaps to sort |

---

## CUDA Concepts Covered

### 1. Sorting Network for 3 Elements

Three elements require exactly **3 comparisons** to sort:

```cuda
// Comparison 1: Sort positions 0 and 1
if (data[0] > data[1]) swap(data[0], data[1]);

// Comparison 2: Sort positions 1 and 2
if (data[1] > data[2]) swap(data[1], data[2]);

// Comparison 3: Sort positions 0 and 1 again
if (data[0] > data[1]) swap(data[0], data[1]);
```

This is known as a **sorting network** — a fixed sequence of compare-swap operations.

### 2. Why 3 Comparisons?

```
Sorting Network Diagram for 3 elements:

    [0]─────●─────────●─────
            │         │
    [1]─────●────●────●─────
                 │
    [2]──────────●──────────

Step 1: Compare (0,1)
Step 2: Compare (1,2)  
Step 3: Compare (0,1)

This guarantees: data[0] ≤ data[1] ≤ data[2]
```

### 3. Step-by-Step Execution

```
Input: [3, 1, 2]

Step 1: Compare data[0] and data[1]
┌───┬───┬───┐
│ 3 │ 1 │ 2 │  →  3 > 1? Yes, swap
└───┴───┴───┘
       ↓
┌───┬───┬───┐
│ 1 │ 3 │ 2 │
└───┴───┴───┘

Step 2: Compare data[1] and data[2]
┌───┬───┬───┐
│ 1 │ 3 │ 2 │  →  3 > 2? Yes, swap
└───┴───┴───┘
       ↓
┌───┬───┬───┐
│ 1 │ 2 │ 3 │
└───┴───┴───┘

Step 3: Compare data[0] and data[1]
┌───┬───┬───┐
│ 1 │ 2 │ 3 │  →  1 > 2? No, no swap
└───┴───┴───┘

Output: 1 2 3 ✓
```

### 4. Handling Duplicate Values

The algorithm correctly handles duplicates (Example 2: `7 7 3`):

```
Input: [7, 7, 3]

Step 1: 7 > 7? No  →  [7, 7, 3]
Step 2: 7 > 3? Yes →  [7, 3, 7]
Step 3: 7 > 3? Yes →  [3, 7, 7]

Output: 3 7 7 ✓
```

### 5. Device Function for Clean Code

Using a helper function makes the code more readable:

```cuda
__device__ void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

__global__ void sortThreeKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        if (data[0] > data[1]) swap(&data[0], &data[1]);
        if (data[1] > data[2]) swap(&data[1], &data[2]);
        if (data[0] > data[1]) swap(&data[0], &data[1]);
    }
}
```

---

## Alternative Solutions

### Using Selection Sort Logic

```cuda
__global__ void sortThreeSelection(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // Find minimum and place at position 0
        int minIdx = 0;
        if (data[1] < data[minIdx]) minIdx = 1;
        if (data[2] < data[minIdx]) minIdx = 2;
        if (minIdx != 0) {
            int temp = data[0];
            data[0] = data[minIdx];
            data[minIdx] = temp;
        }
        
        // Sort remaining two elements
        if (data[1] > data[2]) {
            int temp = data[1];
            data[1] = data[2];
            data[2] = temp;
        }
    }
}
```

### Using Conditional Assignment

```cuda
__global__ void sortThreeConditional(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int a = data[0], b = data[1], c = data[2];
        int min_val, mid_val, max_val;
        
        if (a <= b && a <= c) {
            min_val = a;
            mid_val = (b <= c) ? b : c;
            max_val = (b > c) ? b : c;
        } else if (b <= a && b <= c) {
            min_val = b;
            mid_val = (a <= c) ? a : c;
            max_val = (a > c) ? a : c;
        } else {
            min_val = c;
            mid_val = (a <= b) ? a : b;
            max_val = (a > b) ? a : b;
        }
        
        data[0] = min_val;
        data[1] = mid_val;
        data[2] = max_val;
    }
}
```

---

## Common Mistakes

### ❌ Not Enough Comparisons
```cuda
// Only 2 comparisons — doesn't handle all cases!
if (data[0] > data[1]) swap(data[0], data[1]);
if (data[1] > data[2]) swap(data[1], data[2]);
// Missing third comparison!
// Input [3,1,2] → [1,3,2] → [1,2,3] ✓
// Input [3,2,1] → [2,3,1] → [2,1,3] ✗ (wrong!)
```

### ❌ Wrong Comparison Order
```cuda
// This order doesn't guarantee sorted output
if (data[0] > data[2]) swap(data[0], data[2]);
if (data[0] > data[1]) swap(data[0], data[1]);
// Missing comparison for positions 1 and 2
```

### ❌ Comparing Wrong Indices
```cuda
if (data[0] > data[1]) swap(data[0], data[1]);
if (data[1] > data[2]) swap(data[1], data[2]);
if (data[1] > data[2]) swap(data[1], data[2]);  // Wrong! Should be (0,1)
```

### ❌ Off-by-One Array Access
```cuda
int h_data[3] = {a, b, c};
// ...
printf("%d %d %d\n", h_data[1], h_data[2], h_data[3]);  // h_data[3] is out of bounds!
```

---

## Sorting Complexity Comparison

| Elements | Min Comparisons | Typical Algorithm |
|----------|-----------------|-------------------|
| 2 | 1 | Compare & Swap |
| 3 | 3 | Sorting Network |
| 4 | 5 | Sorting Network |
| N | O(N log N) | Merge/Quick Sort |
| N (parallel) | O(log² N) | Bitonic Sort |

For small fixed-size inputs, sorting networks are optimal.

---

## Key Takeaways

1. **Sorting networks** provide optimal comparison count for small arrays
2. **Three comparisons** are necessary and sufficient for 3 elements
3. **Order matters** — the sequence (0,1), (1,2), (0,1) works correctly
4. **Duplicate handling** is automatic with comparison-based sorting
5. **Building block** for understanding parallel sorting algorithms

---

## Practice Exercises

1. Modify to sort in **descending order**
2. Extend to sort **four** numbers (hint: 5 comparisons needed)
3. Implement using a **parallel approach** with 3 threads
4. What's the minimum comparisons needed for 5 elements? (answer: 7)

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/12)*
