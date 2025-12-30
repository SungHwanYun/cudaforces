# Swap Values by Index

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given three integers and two indices. Swap the values located at the given indices, then output all three integers in order.

### Input
The first line contains three integers: a₁, a₂, and a₃.

The second line contains two integers i and j, representing the positions to swap (1-indexed).

### Output
Print the values of a₁, a₂, and a₃ after performing the swap, separated by spaces.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 1 2 3<br>2 1 | 2 1 3 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 5 10 15<br>1 3 | 15 10 5 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 7 8 9<br>2 3 | 7 9 8 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void swapByIndexKernel(int* data, int i, int j) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
}

int main() {
    int a1, a2, a3;
    int i, j;
    
    scanf("%d %d %d", &a1, &a2, &a3);
    scanf("%d %d", &i, &j);
    
    // Convert to 0-indexed
    i--;
    j--;
    
    // Host array
    int h_data[3] = {a1, a2, a3};
    
    // Device memory
    int* d_data;
    cudaMalloc(&d_data, 3 * sizeof(int));
    
    // Copy to device
    cudaMemcpy(d_data, h_data, 3 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    swapByIndexKernel<<<1, 1>>>(d_data, i, j);
    
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
| ✅ Kernel exists | `__global__ void swapByIndexKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs index-based swap on GPU memory |

---

## CUDA Concepts Covered

### 1. Passing Multiple Parameters to Kernel

Kernels can receive both pointers and scalar values:

```cuda
__global__ void swapByIndexKernel(int* data, int i, int j) {
    //                            ↑         ↑     ↑
    //                         pointer   scalar scalar
    //                         (array)   (index)(index)
}
```

### 2. Index Conversion (1-indexed to 0-indexed)

User input is often 1-indexed, but arrays are 0-indexed:

```cuda
scanf("%d %d", &i, &j);
i--;  // Convert 1-indexed to 0-indexed
j--;  // User's "position 1" → array index 0
```

### 3. Dynamic Index Access in Kernel

The kernel accesses array elements using runtime indices:

```cuda
__global__ void swapByIndexKernel(int* data, int i, int j) {
    if (idx == 0) {
        int temp = data[i];   // i is determined at runtime
        data[i] = data[j];    // Not hardcoded like data[0], data[1]
        data[j] = temp;
    }
}
```

### 4. Memory Layout Visualization

```
Input: a1=5, a2=10, a3=15, swap positions 1 and 3

Host Array (before):
┌─────┬─────┬─────┐
│  5  │ 10  │ 15  │
└─────┴─────┴─────┘
  [0]   [1]   [2]    ← 0-indexed
   ↑           ↑
   i=0        j=2    ← After i--, j--

       cudaMemcpy (HostToDevice)
              ↓

Device Array:
┌─────┬─────┬─────┐
│  5  │ 10  │ 15  │  → Kernel swaps [0] and [2]
└─────┴─────┴─────┘
              ↓
┌─────┬─────┬─────┐
│ 15  │ 10  │  5  │
└─────┴─────┴─────┘

       cudaMemcpy (DeviceToHost)
              ↓

Host Array (after):
┌─────┬─────┬─────┐
│ 15  │ 10  │  5  │
└─────┴─────┴─────┘

Output: 15 10 5
```

### 5. Scalar Parameters vs Pointer Parameters

Understanding when to use each:

```cuda
// Pointer parameter: data that needs to be modified and returned
int* d_data;
cudaMalloc(&d_data, size);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// Scalar parameters: read-only values (indices, sizes, etc.)
swapByIndexKernel<<<1, 1>>>(d_data, i, j);
//                          ↑       ↑  ↑
//                     needs copy  passed by value
//                     back        (no copy back needed)
```

---

## Common Mistakes

### ❌ Forgetting Index Conversion
```cuda
scanf("%d %d", &i, &j);
// Missing: i--; j--;
swapByIndexKernel<<<1, 1>>>(d_data, i, j);  // Off-by-one error!
```
If user inputs `1 3`, you'd access `data[1]` and `data[3]` (out of bounds!).

### ❌ Hardcoding Indices
```cuda
__global__ void swapKernel(int* data) {
    // This only swaps positions 0 and 1, ignoring input indices
    int temp = data[0];
    data[0] = data[1];
    data[1] = temp;
}
```

### ❌ Swapping Indices Instead of Values
```cuda
__global__ void swapByIndexKernel(int* data, int i, int j) {
    if (idx == 0) {
        // Wrong: swapping the index variables, not array elements
        int temp = i;
        i = j;
        j = temp;
    }
}
```

### ❌ Not Copying Updated Data Back
```cuda
swapByIndexKernel<<<1, 1>>>(d_data, i, j);
// Missing cudaMemcpy DeviceToHost!
printf("%d %d %d\n", h_data[0], h_data[1], h_data[2]);  // Still original!
```

### ❌ Array Out of Bounds
```cuda
// If user inputs invalid indices like "4 5" for a 3-element array
// No bounds checking → undefined behavior
```

---

## Key Takeaways

1. **Index conversion** is crucial when user input is 1-indexed
2. **Scalar parameters** (like indices) are passed by value to kernels
3. **Dynamic indexing** allows flexible array access at runtime
4. **Pointer parameters** are used for data that needs modification
5. **Complete validation** should include bounds checking in production code

---

## Practice Exercises

1. Add bounds checking to ensure i and j are valid (1 ≤ i, j ≤ 3)
2. Extend to handle N integers instead of just 3
3. Implement multiple swaps in sequence (given pairs of indices)
4. What happens if i equals j? Optimize for this case

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/8)*
