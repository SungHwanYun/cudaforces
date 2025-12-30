# Sort Three Numbers Descending

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Given three integers, output them in **descending order**.

### Input
Three integers are given in a single line, separated by spaces.

### Output
Print the three integers in descending order, separated by spaces.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 3 1 2 | 3 2 1 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 7 3 7 | 7 7 3 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 3 2 1 | 3 2 1 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void sortThreeDescKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // Sorting network with reversed comparisons
        if (data[0] < data[1]) {
            int temp = data[0];
            data[0] = data[1];
            data[1] = temp;
        }
        if (data[1] < data[2]) {
            int temp = data[1];
            data[1] = data[2];
            data[2] = temp;
        }
        if (data[0] < data[1]) {
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
    sortThreeDescKernel<<<1, 1>>>(d_data);
    
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
| ✅ Kernel exists | `__global__ void sortThreeDescKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Three comparison-swaps for descending order |

---

## CUDA Concepts Covered

### 1. Ascending vs Descending: Flip All Comparisons

The only change from ascending sort is the comparison operator:

```cuda
// Ascending order: swap if a > b (move smaller left)
if (data[i] > data[j]) swap();

// Descending order: swap if a < b (move larger left)
if (data[i] < data[j]) swap();
```

### 2. Descending Sorting Network

Same structure, opposite comparisons:

```
Descending Sorting Network for 3 elements:

    [0]─────●─────────●─────   (larger values bubble left)
            │         │
    [1]─────●────●────●─────
                 │
    [2]──────────●──────────

Step 1: Compare (0,1) — swap if data[0] < data[1]
Step 2: Compare (1,2) — swap if data[1] < data[2]
Step 3: Compare (0,1) — swap if data[0] < data[1]

Result: data[0] ≥ data[1] ≥ data[2]
```

### 3. Step-by-Step Execution

```
Input: [3, 1, 2]

Step 1: Compare data[0] and data[1]
┌───┬───┬───┐
│ 3 │ 1 │ 2 │  →  3 < 1? No, no swap
└───┴───┴───┘

Step 2: Compare data[1] and data[2]
┌───┬───┬───┐
│ 3 │ 1 │ 2 │  →  1 < 2? Yes, swap
└───┴───┴───┘
       ↓
┌───┬───┬───┐
│ 3 │ 2 │ 1 │
└───┴───┴───┘

Step 3: Compare data[0] and data[1]
┌───┬───┬───┐
│ 3 │ 2 │ 1 │  →  3 < 2? No, no swap
└───┴───┴───┘

Output: 3 2 1 ✓
```

### 4. Handling Duplicate Values

Example 2: `7 3 7` → `7 7 3`

```
Input: [7, 3, 7]

Step 1: 7 < 3? No   →  [7, 3, 7]
Step 2: 3 < 7? Yes  →  [7, 7, 3]
Step 3: 7 < 7? No   →  [7, 7, 3]

Output: 7 7 3 ✓
```

### 5. Already Sorted Input

Example 3: `3 2 1` — already in descending order:

```
Input: [3, 2, 1]

Step 1: 3 < 2? No  →  [3, 2, 1]
Step 2: 2 < 1? No  →  [3, 2, 1]
Step 3: 3 < 2? No  →  [3, 2, 1]

Output: 3 2 1 ✓ (no swaps needed)
```

---

## Comparison: Ascending vs Descending

| Aspect | Ascending | Descending |
|--------|-----------|------------|
| Swap condition | `a > b` | `a < b` |
| Result order | `[min, mid, max]` | `[max, mid, min]` |
| Example `[3,1,2]` | `[1, 2, 3]` | `[3, 2, 1]` |

The **same sorting network** structure works — just invert all comparisons.

---

## Alternative Solutions

### Using Device Swap Function

```cuda
__device__ void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

__global__ void sortThreeDescKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        if (data[0] < data[1]) swap(&data[0], &data[1]);
        if (data[1] < data[2]) swap(&data[1], &data[2]);
        if (data[0] < data[1]) swap(&data[0], &data[1]);
    }
}
```

### Using max/min Logic

```cuda
__global__ void sortThreeDescKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int a = data[0], b = data[1], c = data[2];
        
        // Find max, mid, min
        int max_val = a;
        if (b > max_val) max_val = b;
        if (c > max_val) max_val = c;
        
        int min_val = a;
        if (b < min_val) min_val = b;
        if (c < min_val) min_val = c;
        
        int mid_val = a + b + c - max_val - min_val;
        
        data[0] = max_val;
        data[1] = mid_val;
        data[2] = min_val;
    }
}
```

### Parameterized Sort Direction

```cuda
__global__ void sortThreeKernel(int* data, int descending) {
    int idx = threadIdx.x;
    if (idx == 0) {
        for (int pass = 0; pass < 2; pass++) {
            if (descending ? (data[0] < data[1]) : (data[0] > data[1])) {
                int temp = data[0]; data[0] = data[1]; data[1] = temp;
            }
            if (descending ? (data[1] < data[2]) : (data[1] > data[2])) {
                int temp = data[1]; data[1] = data[2]; data[2] = temp;
            }
        }
    }
}
```

---

## Common Mistakes

### ❌ Using Ascending Comparison
```cuda
// This sorts in ASCENDING order!
if (data[0] > data[1]) swap();  // Wrong for descending!
```
Use `<` instead of `>` for descending order.

### ❌ Mixing Comparison Operators
```cuda
if (data[0] < data[1]) swap();  // Correct
if (data[1] > data[2]) swap();  // Wrong! Should be <
if (data[0] < data[1]) swap();  // Correct
```
All three comparisons must use the same operator direction.

### ❌ Forgetting Third Comparison
```cuda
if (data[0] < data[1]) swap(data[0], data[1]);
if (data[1] < data[2]) swap(data[1], data[2]);
// Missing! Input [1,3,2] → [3,1,2] → [3,2,1] ✓
// But: Input [1,2,3] → [2,1,3] → [2,3,1] ✗
```

---

## Relationship to Previous Problems

| Problem | Elements | Comparisons | Condition |
|---------|----------|-------------|-----------|
| Sort Two Numbers | 2 | 1 | `a > b` |
| Sort Two Desc | 2 | 1 | `a < b` |
| Sort Three Numbers | 3 | 3 | `a > b` |
| **Sort Three Desc** | 3 | 3 | `a < b` |

The pattern: flip `>` to `<` to change from ascending to descending.

---

## Key Takeaways

1. **Same network, opposite comparisons** — descending just flips `>` to `<`
2. **Three comparisons** remain optimal for 3 elements
3. **Sorting network structure** is direction-agnostic
4. **Duplicates handled** automatically by comparison-based sorting
5. **Pattern recognition** — ascending ↔ descending conversion is consistent

---

## Practice Exercises

1. Create a single kernel with a `direction` parameter for both orders
2. Extend to sort **four** numbers in descending order
3. Test with negative numbers (e.g., `-5, 0, 3`)
4. Implement parallel version where each comparison is a separate thread

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/13)*
