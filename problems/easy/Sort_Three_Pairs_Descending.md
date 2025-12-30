# Sort Three Pairs Descending

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given three pairs (a₁, b₁), (a₂, b₂), and (a₃, b₃). Sort them in **descending order**.

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
Print the three pairs in descending order, one per line.

Each line should contain the first element and second element separated by a space.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 3 5<br>1 4<br>2 8 | 3 5<br>2 8<br>1 4 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 5 10<br>5 3<br>5 7 | 5 10<br>5 7<br>5 3 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 1 1<br>2 2<br>3 3 | 3 3<br>2 2<br>1 1 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__device__ int pairLess(int a1, int b1, int a2, int b2) {
    // Returns 1 if (a1, b1) < (a2, b2) lexicographically
    if (a1 < a2) return 1;
    if (a1 == a2 && b1 < b2) return 1;
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

__global__ void sortThreePairsDescKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // Sorting network for 3 elements: 3 comparisons (descending)
        // Compare pair 0 and pair 1
        if (pairLess(data[0], data[1], data[2], data[3])) {
            swapPairs(data, 0, 1);
        }
        // Compare pair 1 and pair 2
        if (pairLess(data[2], data[3], data[4], data[5])) {
            swapPairs(data, 1, 2);
        }
        // Compare pair 0 and pair 1 again
        if (pairLess(data[0], data[1], data[2], data[3])) {
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
    sortThreePairsDescKernel<<<1, 1>>>(d_data);
    
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
| ✅ Kernel exists | `__global__ void sortThreePairsDescKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Sorting network with descending lexicographic comparison |

---

## CUDA Concepts Covered

### 1. Ascending vs Descending: Only Change Comparison

The only difference from ascending Sort Three Pairs:

```cuda
// Ascending: swap if pair_i > pair_j (pairGreater)
if (pairGreater(...)) { swapPairs(...); }

// Descending: swap if pair_i < pair_j (pairLess)
if (pairLess(...)) { swapPairs(...); }
```

The sorting network structure remains **exactly the same**.

### 2. Descending Sorting Network

```
Descending Sorting Network for 3 pairs:

Pair 0 ─────●─────────●─────  (larger pairs bubble to top)
            │         │
Pair 1 ─────●────●────●─────
                 │
Pair 2 ──────────●──────────

Step 1: Compare pairs (0, 1) — swap if pair0 < pair1
Step 2: Compare pairs (1, 2) — swap if pair1 < pair2
Step 3: Compare pairs (0, 1) — swap if pair0 < pair1

Result: pair[0] ≥ pair[1] ≥ pair[2] (lexicographically)
```

### 3. Example Walkthrough

**Example 1**: `(3, 5)`, `(1, 4)`, `(2, 8)`

```
Initial: [(3,5), (1,4), (2,8)]

Step 1: Compare (3,5) vs (1,4)
  3 < 1? No → NO SWAP
  Result: [(3,5), (1,4), (2,8)]

Step 2: Compare (1,4) vs (2,8)
  1 < 2? Yes → SWAP
  Result: [(3,5), (2,8), (1,4)]

Step 3: Compare (3,5) vs (2,8)
  3 < 2? No → NO SWAP
  Result: [(3,5), (2,8), (1,4)]

Output:
3 5
2 8
1 4
```

**Example 2**: `(5, 10)`, `(5, 3)`, `(5, 7)` — all have same first element

```
Initial: [(5,10), (5,3), (5,7)]

Step 1: Compare (5,10) vs (5,3)
  5 < 5? No
  5 == 5 && 10 < 3? No → NO SWAP
  Result: [(5,10), (5,3), (5,7)]

Step 2: Compare (5,3) vs (5,7)
  5 < 5? No
  5 == 5 && 3 < 7? Yes → SWAP
  Result: [(5,10), (5,7), (5,3)]

Step 3: Compare (5,10) vs (5,7)
  5 < 5? No
  5 == 5 && 10 < 7? No → NO SWAP
  Result: [(5,10), (5,7), (5,3)]

Output:
5 10
5 7
5 3
```

**Example 3**: `(1, 1)`, `(2, 2)`, `(3, 3)` — ascending input, need full reversal

```
Initial: [(1,1), (2,2), (3,3)]

Step 1: Compare (1,1) vs (2,2)
  1 < 2? Yes → SWAP
  Result: [(2,2), (1,1), (3,3)]

Step 2: Compare (1,1) vs (3,3)
  1 < 3? Yes → SWAP
  Result: [(2,2), (3,3), (1,1)]

Step 3: Compare (2,2) vs (3,3)
  2 < 3? Yes → SWAP
  Result: [(3,3), (2,2), (1,1)]

Output:
3 3
2 2
1 1
```

### 4. Comparison Function: pairLess vs pairGreater

```cuda
// For ASCENDING sort
__device__ int pairGreater(int a1, int b1, int a2, int b2) {
    if (a1 > a2) return 1;
    if (a1 == a2 && b1 > b2) return 1;
    return 0;
}

// For DESCENDING sort
__device__ int pairLess(int a1, int b1, int a2, int b2) {
    if (a1 < a2) return 1;
    if (a1 == a2 && b1 < b2) return 1;
    return 0;
}
```

### 5. Memory Layout Reminder

```
Array index:  [0]   [1]   [2]   [3]   [4]   [5]
Content:       a1    b1    a2    b2    a3    b3
              └─pair 0─┘  └─pair 1─┘  └─pair 2─┘

Pair i: data[i*2] = a, data[i*2+1] = b
```

---

## Comparison: Ascending vs Descending

| Aspect | Sort Three Pairs (Asc) | Sort Three Pairs (Desc) |
|--------|------------------------|-------------------------|
| Comparison function | `pairGreater` | `pairLess` |
| Swap condition | `pair_i > pair_j` | `pair_i < pair_j` |
| Output order | Smallest → Largest | Largest → Smallest |
| Network structure | Same (3 comparisons) | Same (3 comparisons) |

---

## Alternative Solutions

### Inline Comparison (No Helper Functions)

```cuda
__global__ void sortThreePairsDescKernel(int* d) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // Compare pairs 0, 1
        if (d[0] < d[2] || (d[0] == d[2] && d[1] < d[3])) {
            int t0 = d[0], t1 = d[1];
            d[0] = d[2]; d[1] = d[3];
            d[2] = t0; d[3] = t1;
        }
        // Compare pairs 1, 2
        if (d[2] < d[4] || (d[2] == d[4] && d[3] < d[5])) {
            int t0 = d[2], t1 = d[3];
            d[2] = d[4]; d[3] = d[5];
            d[4] = t0; d[5] = t1;
        }
        // Compare pairs 0, 1
        if (d[0] < d[2] || (d[0] == d[2] && d[1] < d[3])) {
            int t0 = d[0], t1 = d[1];
            d[0] = d[2]; d[1] = d[3];
            d[2] = t0; d[3] = t1;
        }
    }
}
```

### Parameterized Sort Direction

```cuda
__device__ int comparePairs(int a1, int b1, int a2, int b2, int descending) {
    // Returns 1 if should swap based on direction
    if (descending) {
        // Descending: swap if pair1 < pair2
        if (a1 < a2) return 1;
        if (a1 == a2 && b1 < b2) return 1;
    } else {
        // Ascending: swap if pair1 > pair2
        if (a1 > a2) return 1;
        if (a1 == a2 && b1 > b2) return 1;
    }
    return 0;
}

__global__ void sortThreePairsKernel(int* data, int descending) {
    int idx = threadIdx.x;
    if (idx == 0) {
        if (comparePairs(data[0], data[1], data[2], data[3], descending))
            swapPairs(data, 0, 1);
        if (comparePairs(data[2], data[3], data[4], data[5], descending))
            swapPairs(data, 1, 2);
        if (comparePairs(data[0], data[1], data[2], data[3], descending))
            swapPairs(data, 0, 1);
    }
}
```

---

## Common Mistakes

### ❌ Using Ascending Comparison
```cuda
// This sorts in ASCENDING order!
if (pairGreater(data[0], data[1], data[2], data[3])) {
    swapPairs(data, 0, 1);
}
```
Use `pairLess` (or flip operators) for descending.

### ❌ Forgetting Third Comparison
```cuda
if (pairLess(...)) swapPairs(data, 0, 1);
if (pairLess(...)) swapPairs(data, 1, 2);
// Missing third comparison!
```
Three comparisons are required to sort 3 elements.

### ❌ Incomplete Pair Swap
```cuda
// Only swapping first element of each pair
int temp = data[i * 2];
data[i * 2] = data[j * 2];
data[j * 2] = temp;
// Second element not swapped!
```

---

## Complete Problem Series Summary

| Problem | Elements | Direction | Comparisons | Key Concept |
|---------|----------|-----------|-------------|-------------|
| Sort Two Numbers | 2 ints | Asc | 1 | Basic swap |
| Sort Two Numbers Desc | 2 ints | Desc | 1 | Flip operator |
| Sort Three Numbers | 3 ints | Asc | 3 | Sorting network |
| Sort Three Numbers Desc | 3 ints | Desc | 3 | Network + flip |
| Sort Two Pairs | 2 pairs | Asc | 1 | Lexicographic |
| Sort Two Pairs Desc | 2 pairs | Desc | 1 | Lex + flip |
| Sort Three Pairs | 3 pairs | Asc | 3 | Network + lex |
| **Sort Three Pairs Desc** | 3 pairs | Desc | 3 | All combined |

This problem represents the culmination of all sorting concepts covered!

---

## Key Takeaways

1. **Same sorting network** — only comparison direction changes
2. **`pairLess` instead of `pairGreater`** — flip `>` to `<`
3. **Three comparisons** still required for 3 elements
4. **Consistent pattern** — ascending ↔ descending conversion
5. **Building blocks combine** — network + lexicographic + direction

---

## Practice Exercises

1. Sort **four pairs** in descending order (5 comparisons)
2. Create a **unified kernel** that handles both directions with a parameter
3. Extend pairs to **triples** (a, b, c) with 3-level descending comparison
4. Implement using **parallel odd-even transposition sort**

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/17)*
