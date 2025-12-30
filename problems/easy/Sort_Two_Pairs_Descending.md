# Sort Two Pairs Descending

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given two pairs (a₁, b₁) and (a₂, b₂). Sort them in **descending order**.

A pair (aᵢ, bᵢ) is considered **smaller** than (aⱼ, bⱼ) if:
- aᵢ < aⱼ, **or**
- aᵢ = aⱼ and bᵢ < bⱼ

This is called **lexicographic ordering** (dictionary order).

### Input
The first line contains two integers a₁ and b₁ representing the first pair.

The second line contains two integers a₂ and b₂ representing the second pair.

It is guaranteed that 1 ≤ a₁, b₁, a₂, b₂ ≤ 100 and (a₁, b₁) ≠ (a₂, b₂).

### Output
Print the **larger** pair on the first line and the **smaller** pair on the second line.

Each line should contain the first element and second element separated by a space.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 1 4<br>3 5 | 3 5<br>1 4 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 5 3<br>5 10 | 5 10<br>5 3 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 2 1<br>1 2 | 2 1<br>1 2 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void sortPairsDescKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int a1 = data[0], b1 = data[1];
        int a2 = data[2], b2 = data[3];
        
        // Lexicographic comparison: swap if pair1 < pair2
        int shouldSwap = 0;
        if (a1 < a2) {
            shouldSwap = 1;
        } else if (a1 == a2 && b1 < b2) {
            shouldSwap = 1;
        }
        
        if (shouldSwap) {
            data[0] = a2; data[1] = b2;
            data[2] = a1; data[3] = b1;
        }
    }
}

int main() {
    int a1, b1, a2, b2;
    scanf("%d %d", &a1, &b1);
    scanf("%d %d", &a2, &b2);
    
    // Host array: [a1, b1, a2, b2]
    int h_data[4] = {a1, b1, a2, b2};
    
    // Device memory
    int* d_data;
    cudaMalloc(&d_data, 4 * sizeof(int));
    
    // Copy to device
    cudaMemcpy(d_data, h_data, 4 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    sortPairsDescKernel<<<1, 1>>>(d_data);
    
    // Copy back to host
    cudaMemcpy(h_data, d_data, 4 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d %d\n", h_data[0], h_data[1]);
    printf("%d %d\n", h_data[2], h_data[3]);
    
    cudaFree(d_data);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void sortPairsDescKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Lexicographic comparison for descending order |

---

## CUDA Concepts Covered

### 1. Ascending vs Descending: Flip the Comparison

The only change from ascending sort is the comparison operator:

```cuda
// Ascending order: swap if pair1 > pair2
if (a1 > a2 || (a1 == a2 && b1 > b2)) { swap(); }

// Descending order: swap if pair1 < pair2
if (a1 < a2 || (a1 == a2 && b1 < b2)) { swap(); }
```

### 2. Descending Lexicographic Comparison

```
Descending order means: larger pair comes first

Compare pair1 vs pair2:
  - If a1 < a2: pair1 is smaller → swap to put pair2 first
  - If a1 == a2 and b1 < b2: pair1 is smaller → swap to put pair2 first
  - Otherwise: pair1 >= pair2 → no swap needed
```

### 3. Example Walkthrough

**Example 1**: `(1, 4)` and `(3, 5)`

```
a1=1, b1=4, a2=3, b2=5

Compare a1 vs a2: 1 < 3? Yes → shouldSwap = 1

After swap:
  data[0]=3, data[1]=5  → (3, 5) - larger
  data[2]=1, data[3]=4  → (1, 4) - smaller

Output:
3 5
1 4
```

**Example 2**: `(5, 3)` and `(5, 10)`

```
a1=5, b1=3, a2=5, b2=10

Compare a1 vs a2: 5 < 5? No
Compare a1 == a2: 5 == 5? Yes
Compare b1 vs b2: 3 < 10? Yes → shouldSwap = 1

After swap:
  data[0]=5, data[1]=10  → (5, 10) - larger
  data[2]=5, data[3]=3   → (5, 3) - smaller

Output:
5 10
5 3
```

**Example 3**: `(2, 1)` and `(1, 2)`

```
a1=2, b1=1, a2=1, b2=2

Compare a1 vs a2: 2 < 1? No
Compare a1 == a2: 2 == 1? No → shouldSwap = 0

No swap needed, pair1 is already larger.

Output:
2 1
1 2
```

### 4. Comparison Table: Ascending vs Descending

| Aspect | Ascending | Descending |
|--------|-----------|------------|
| Swap condition | `pair1 > pair2` | `pair1 < pair2` |
| First output | Smaller pair | Larger pair |
| Second output | Larger pair | Smaller pair |
| Comparison operators | `>` | `<` |

### 5. Decision Tree Visualization

```
      a1 vs a2
      /      \
   a1 < a2   a1 >= a2
     |          |
   SWAP     a1 == a2?
              /    \
            Yes     No
             |       |
         b1 < b2?  NO SWAP (a1 > a2, pair1 is larger)
           /   \
         Yes   No
          |     |
        SWAP  NO SWAP
```

---

## Comparison with Ascending Version

| Sort Two Pairs (Ascending) | Sort Two Pairs Descending |
|---------------------------|---------------------------|
| `if (a1 > a2)` | `if (a1 < a2)` |
| `if (a1 == a2 && b1 > b2)` | `if (a1 == a2 && b1 < b2)` |
| Output: smaller, larger | Output: larger, smaller |

**Same structure, opposite comparisons.**

---

## Alternative Solutions

### Using Combined Condition

```cuda
__global__ void sortPairsDescKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int a1 = data[0], b1 = data[1];
        int a2 = data[2], b2 = data[3];
        
        // Single combined condition
        if (a1 < a2 || (a1 == a2 && b1 < b2)) {
            data[0] = a2; data[1] = b2;
            data[2] = a1; data[3] = b1;
        }
    }
}
```

### Using Device Comparison Function

```cuda
__device__ int pairLess(int a1, int b1, int a2, int b2) {
    // Returns 1 if (a1, b1) < (a2, b2) lexicographically
    if (a1 < a2) return 1;
    if (a1 == a2 && b1 < b2) return 1;
    return 0;
}

__global__ void sortPairsDescKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        if (pairLess(data[0], data[1], data[2], data[3])) {
            // Swap pairs
            int t0 = data[0], t1 = data[1];
            data[0] = data[2]; data[1] = data[3];
            data[2] = t0; data[3] = t1;
        }
    }
}
```

### Parameterized Sort Direction

```cuda
__global__ void sortPairsKernel(int* data, int descending) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int a1 = data[0], b1 = data[1];
        int a2 = data[2], b2 = data[3];
        
        int shouldSwap;
        if (descending) {
            // Descending: swap if pair1 < pair2
            shouldSwap = (a1 < a2) || (a1 == a2 && b1 < b2);
        } else {
            // Ascending: swap if pair1 > pair2
            shouldSwap = (a1 > a2) || (a1 == a2 && b1 > b2);
        }
        
        if (shouldSwap) {
            data[0] = a2; data[1] = b2;
            data[2] = a1; data[3] = b1;
        }
    }
}
```

---

## Common Mistakes

### ❌ Using Ascending Comparison
```cuda
// This sorts in ASCENDING order!
if (a1 > a2 || (a1 == a2 && b1 > b2)) {
    // swap
}
```
Use `<` instead of `>` for descending order.

### ❌ Mixing Operators
```cuda
// Inconsistent comparison operators
if (a1 < a2 || (a1 == a2 && b1 > b2)) {  // Wrong! Mixed < and >
    // swap
}
```
Both comparisons must use the same direction (`<` for descending).

### ❌ Swapping Only Part of Pair
```cuda
if (shouldSwap) {
    int temp = data[0];
    data[0] = data[2];
    data[2] = temp;
    // Missing: b values not swapped!
}
```

---

## Relationship to Other Problems

| Problem | Direction | Swap When |
|---------|-----------|-----------|
| Sort Two Pairs | Ascending | `pair1 > pair2` |
| **Sort Two Pairs Desc** | Descending | `pair1 < pair2` |
| Sort Three Pairs | Ascending | `pair1 > pair2` (3×) |
| Sort Three Pairs Desc | Descending | `pair1 < pair2` (3×) |

The pattern is consistent: flip `>` to `<` for descending order.

---

## Key Takeaways

1. **Flip comparison operators** — `>` becomes `<` for descending
2. **Same lexicographic logic** — compare first element, then second
3. **Larger pair first** in descending order output
4. **Swap entire pairs** — both elements must move together
5. **Consistent pattern** with number sorting (ascending ↔ descending)

---

## Practice Exercises

1. Create a single kernel with a `direction` parameter
2. Extend to sort **three pairs** in descending order
3. Test with edge cases: `(1, 100)` vs `(100, 1)`
4. Implement using a generic comparison function

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/16)*
