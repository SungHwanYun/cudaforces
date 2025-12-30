# Sort Three Pairs

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given three pairs (a₁, b₁), (a₂, b₂), and (a₃, b₃). Sort them in ascending order.

A pair (aᵢ, bᵢ) is considered **smaller** than (aⱼ, bⱼ) if:
- aᵢ < aⱼ, **or**
- aᵢ = aⱼ and bᵢ < bⱼ

This is called **lexicographic ordering** (dictionary order).

### Input
The first line contains two integers a₁ and b₁ representing the first pair.

The second line contains two integers a₂ and b₂ representing the second pair.

The third line contains two integers a₃ and b₃ representing the third pair.

It is guaranteed that 1 ≤ aᵢ, bᵢ ≤ 100 and (aᵢ, bᵢ) ≠ (aⱼ, bⱼ) for all 1 ≤ i ≠ j ≤ 3.

### Output
Print the three pairs in sorted order, one per line.

Each line should contain the first element and second element separated by a space.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 3 5<br>1 4<br>2 8 | 1 4<br>2 8<br>3 5 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 5 10<br>5 3<br>5 7 | 5 3<br>5 7<br>5 10 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 1 2<br>2 1<br>1 1 | 1 1<br>1 2<br>2 1 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__device__ int pairGreater(int a1, int b1, int a2, int b2) {
    // Returns 1 if (a1, b1) > (a2, b2) lexicographically
    if (a1 > a2) return 1;
    if (a1 == a2 && b1 > b2) return 1;
    return 0;
}

__device__ void swapPairs(int* data, int i, int j) {
    // Swap pairs at positions i and j (each pair is 2 elements)
    int tempA = data[i * 2];
    int tempB = data[i * 2 + 1];
    data[i * 2] = data[j * 2];
    data[i * 2 + 1] = data[j * 2 + 1];
    data[j * 2] = tempA;
    data[j * 2 + 1] = tempB;
}

__global__ void sortThreePairsKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // Sorting network for 3 elements: 3 comparisons
        // Compare pair 0 and pair 1
        if (pairGreater(data[0], data[1], data[2], data[3])) {
            swapPairs(data, 0, 1);
        }
        // Compare pair 1 and pair 2
        if (pairGreater(data[2], data[3], data[4], data[5])) {
            swapPairs(data, 1, 2);
        }
        // Compare pair 0 and pair 1 again
        if (pairGreater(data[0], data[1], data[2], data[3])) {
            swapPairs(data, 0, 1);
        }
    }
}

int main() {
    int a1, b1, a2, b2, a3, b3;
    scanf("%d %d", &a1, &b1);
    scanf("%d %d", &a2, &b2);
    scanf("%d %d", &a3, &b3);
    
    // Host array: [a1, b1, a2, b2, a3, b3]
    int h_data[6] = {a1, b1, a2, b2, a3, b3};
    
    // Device memory
    int* d_data;
    cudaMalloc(&d_data, 6 * sizeof(int));
    
    // Copy to device
    cudaMemcpy(d_data, h_data, 6 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    sortThreePairsKernel<<<1, 1>>>(d_data);
    
    // Copy back to host
    cudaMemcpy(h_data, d_data, 6 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d %d\n", h_data[0], h_data[1]);
    printf("%d %d\n", h_data[2], h_data[3]);
    printf("%d %d\n", h_data[4], h_data[5]);
    
    cudaFree(d_data);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void sortThreePairsKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Sorting network with lexicographic comparison |

---

## CUDA Concepts Covered

### 1. Combining Previous Concepts

This problem combines:
- **Sorting Network** from [Sort Three Numbers](Sort_Three_Numbers.md) — 3 comparisons
- **Lexicographic Comparison** from [Sort Two Pairs](Sort_Two_Pairs.md) — compound conditions

```
Sort Three Numbers + Pair Comparison = Sort Three Pairs
```

### 2. Memory Layout for Multiple Pairs

```
Array index:  [0]   [1]   [2]   [3]   [4]   [5]
Content:       a1    b1    a2    b2    a3    b3
              └─pair 0─┘  └─pair 1─┘  └─pair 2─┘
              
Pair i starts at index: i * 2
  - First element:  data[i * 2]
  - Second element: data[i * 2 + 1]
```

### 3. Device Helper Functions

Using `__device__` functions for cleaner code:

```cuda
__device__ int pairGreater(int a1, int b1, int a2, int b2) {
    // Lexicographic comparison
    if (a1 > a2) return 1;
    if (a1 == a2 && b1 > b2) return 1;
    return 0;
}

__device__ void swapPairs(int* data, int i, int j) {
    // Swap pairs by index
    int tempA = data[i * 2];
    int tempB = data[i * 2 + 1];
    data[i * 2] = data[j * 2];
    data[i * 2 + 1] = data[j * 2 + 1];
    data[j * 2] = tempA;
    data[j * 2 + 1] = tempB;
}
```

### 4. Sorting Network for 3 Pairs

Same structure as sorting 3 numbers, but with pair comparisons:

```
Sorting Network:

Pair 0 ─────●─────────●─────
            │         │
Pair 1 ─────●────●────●─────
                 │
Pair 2 ──────────●──────────

Step 1: Compare pairs (0, 1)
Step 2: Compare pairs (1, 2)
Step 3: Compare pairs (0, 1)

Result: pair[0] ≤ pair[1] ≤ pair[2] (lexicographically)
```

### 5. Example Walkthrough

**Example 2**: `(5, 10)`, `(5, 3)`, `(5, 7)` — all have same first element

```
Initial: [(5,10), (5,3), (5,7)]

Step 1: Compare (5,10) vs (5,3)
  5 > 5? No
  5 == 5 && 10 > 3? Yes → SWAP
  Result: [(5,3), (5,10), (5,7)]

Step 2: Compare (5,10) vs (5,7)
  5 > 5? No
  5 == 5 && 10 > 7? Yes → SWAP
  Result: [(5,3), (5,7), (5,10)]

Step 3: Compare (5,3) vs (5,7)
  5 > 5? No
  5 == 5 && 3 > 7? No → NO SWAP
  Result: [(5,3), (5,7), (5,10)]

Output:
5 3
5 7
5 10
```

**Example 3**: `(1, 2)`, `(2, 1)`, `(1, 1)` — mixed first elements

```
Initial: [(1,2), (2,1), (1,1)]

Step 1: Compare (1,2) vs (2,1)
  1 > 2? No → NO SWAP
  Result: [(1,2), (2,1), (1,1)]

Step 2: Compare (2,1) vs (1,1)
  2 > 1? Yes → SWAP
  Result: [(1,2), (1,1), (2,1)]

Step 3: Compare (1,2) vs (1,1)
  1 > 1? No
  1 == 1 && 2 > 1? Yes → SWAP
  Result: [(1,1), (1,2), (2,1)]

Output:
1 1
1 2
2 1
```

---

## Alternative Solutions

### Inline Comparison (No Helper Functions)

```cuda
__global__ void sortThreePairsKernel(int* d) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // Compare and swap pairs 0, 1
        if (d[0] > d[2] || (d[0] == d[2] && d[1] > d[3])) {
            int t0 = d[0], t1 = d[1];
            d[0] = d[2]; d[1] = d[3];
            d[2] = t0; d[3] = t1;
        }
        // Compare and swap pairs 1, 2
        if (d[2] > d[4] || (d[2] == d[4] && d[3] > d[5])) {
            int t0 = d[2], t1 = d[3];
            d[2] = d[4]; d[3] = d[5];
            d[4] = t0; d[5] = t1;
        }
        // Compare and swap pairs 0, 1
        if (d[0] > d[2] || (d[0] == d[2] && d[1] > d[3])) {
            int t0 = d[0], t1 = d[1];
            d[0] = d[2]; d[1] = d[3];
            d[2] = t0; d[3] = t1;
        }
    }
}
```

### Using Comparison Result Integer

```cuda
__device__ int comparePairs(int* data, int i, int j) {
    int ai = data[i * 2], bi = data[i * 2 + 1];
    int aj = data[j * 2], bj = data[j * 2 + 1];
    
    if (ai != aj) return ai - aj;
    return bi - bj;
}

__global__ void sortThreePairsKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        if (comparePairs(data, 0, 1) > 0) swapPairs(data, 0, 1);
        if (comparePairs(data, 1, 2) > 0) swapPairs(data, 1, 2);
        if (comparePairs(data, 0, 1) > 0) swapPairs(data, 0, 1);
    }
}
```

---

## Common Mistakes

### ❌ Wrong Index Calculation
```cuda
// Wrong: treating pair index as array index
if (pairGreater(data[0], data[1], data[1], data[2])) {  // Overlapping!
    // ...
}
```
Pair 0 is at indices `[0, 1]`, Pair 1 is at indices `[2, 3]`.

### ❌ Incomplete Pair Swap
```cuda
// Wrong: swapping only first elements
int temp = data[i * 2];
data[i * 2] = data[j * 2];
data[j * 2] = temp;
// Missing second element swap!
```

### ❌ Only Two Comparisons
```cuda
// Insufficient comparisons - doesn't sort all cases
if (...) swapPairs(data, 0, 1);
if (...) swapPairs(data, 1, 2);
// Missing third comparison!
```

### ❌ Wrong Comparison Order in Lexicographic
```cuda
// Wrong: checking second element first
if (b1 > b2 || (b1 == b2 && a1 > a2)) {
    // This sorts by (b, a) not (a, b)!
}
```

---

## Building Blocks Summary

| Problem | Elements | Comparisons | Key Concept |
|---------|----------|-------------|-------------|
| Sort Two Numbers | 2 ints | 1 | Basic compare-swap |
| Sort Three Numbers | 3 ints | 3 | Sorting network |
| Sort Two Pairs | 2 pairs | 1 | Lexicographic comparison |
| **Sort Three Pairs** | 3 pairs | 3 | Network + Lexicographic |

This problem demonstrates how fundamental concepts combine to solve complex problems.

---

## Key Takeaways

1. **Combine concepts** — sorting network + lexicographic comparison
2. **Index arithmetic** — pair `i` at indices `[i*2, i*2+1]`
3. **`__device__` functions** improve code readability and reusability
4. **Swap entire pairs** — both elements must move together
5. **Scalable pattern** — same approach works for N pairs

---

## Practice Exercises

1. Sort **four pairs** lexicographically (5 comparisons needed)
2. Sort three pairs in **descending** order
3. Extend pairs to **triples** (a, b, c) with 3-level comparison
4. Implement parallel sorting where each comparison is a separate thread

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/15)*
