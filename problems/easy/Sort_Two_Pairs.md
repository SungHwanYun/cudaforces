# Sort Two Pairs

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given two pairs (a₁, b₁) and (a₂, b₂). Sort them in ascending order.

A pair (aᵢ, bᵢ) is considered **smaller** than (aⱼ, bⱼ) if:
- aᵢ < aⱼ, **or**
- aᵢ = aⱼ and bᵢ < bⱼ

This is called **lexicographic ordering** (dictionary order).

### Input
The first line contains two integers a₁ and b₁ representing the first pair.

The second line contains two integers a₂ and b₂ representing the second pair.

It is guaranteed that 1 ≤ a₁, b₁, a₂, b₂ ≤ 100 and (a₁, b₁) ≠ (a₂, b₂).

### Output
Print the smaller pair on the first line and the larger pair on the second line.

Each line should contain the first element and second element separated by a space.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 3 5<br>1 4 | 1 4<br>3 5 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 5 10<br>5 3 | 5 3<br>5 10 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 1 2<br>2 1 | 1 2<br>2 1 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void sortPairsKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int a1 = data[0], b1 = data[1];
        int a2 = data[2], b2 = data[3];
        
        // Lexicographic comparison: compare first element, then second
        int shouldSwap = 0;
        if (a1 > a2) {
            shouldSwap = 1;
        } else if (a1 == a2 && b1 > b2) {
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
    sortPairsKernel<<<1, 1>>>(d_data);
    
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
| ✅ Kernel exists | `__global__ void sortPairsKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Lexicographic comparison and swap |

---

## CUDA Concepts Covered

### 1. Lexicographic (Dictionary) Ordering

Pairs are compared like words in a dictionary:

```
Compare first elements (a):
  - If different → determines order
  - If same → compare second elements (b)

Example: Is (3, 5) < (1, 4)?
  Step 1: Compare a's: 3 vs 1 → 3 > 1
  Result: (3, 5) > (1, 4), so (1, 4) comes first
```

### 2. Compound Comparison Logic

```cuda
int shouldSwap = 0;
if (a1 > a2) {
    shouldSwap = 1;                    // First element decides
} else if (a1 == a2 && b1 > b2) {
    shouldSwap = 1;                    // First equal, second decides
}
```

This can be visualized as:

```
      a1 vs a2
      /      \
   a1 > a2   a1 <= a2
     |          |
   SWAP     a1 == a2?
              /    \
            Yes     No
             |       |
         b1 > b2?  NO SWAP
           /   \
         Yes   No
          |     |
        SWAP  NO SWAP
```

### 3. Memory Layout for Pairs

Storing pairs as contiguous integers:

```
Array index:  [0]   [1]   [2]   [3]
Content:       a1    b1    a2    b2
              └─pair 1─┘  └─pair 2─┘
```

Swapping pairs means swapping both elements:

```cuda
if (shouldSwap) {
    data[0] = a2; data[1] = b2;  // Pair 2 → Position 1
    data[2] = a1; data[3] = b1;  // Pair 1 → Position 2
}
```

### 4. Example Walkthrough

**Example 1**: `(3, 5)` and `(1, 4)`

```
a1=3, b1=5, a2=1, b2=4

Compare a1 vs a2: 3 > 1? Yes → shouldSwap = 1

After swap:
  data[0]=1, data[1]=4  → (1, 4)
  data[2]=3, data[3]=5  → (3, 5)

Output:
1 4
3 5
```

**Example 2**: `(5, 10)` and `(5, 3)`

```
a1=5, b1=10, a2=5, b2=3

Compare a1 vs a2: 5 > 5? No
Compare a1 == a2: 5 == 5? Yes
Compare b1 vs b2: 10 > 3? Yes → shouldSwap = 1

After swap:
  data[0]=5, data[1]=3   → (5, 3)
  data[2]=5, data[3]=10  → (5, 10)

Output:
5 3
5 10
```

**Example 3**: `(1, 2)` and `(2, 1)`

```
a1=1, b1=2, a2=2, b2=1

Compare a1 vs a2: 1 > 2? No
Compare a1 == a2: 1 == 2? No → shouldSwap = 0

No swap needed, already in order.

Output:
1 2
2 1
```

### 5. Alternative: Combined Comparison

Using a single expression for comparison:

```cuda
__global__ void sortPairsKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int a1 = data[0], b1 = data[1];
        int a2 = data[2], b2 = data[3];
        
        // Single comparison: true if pair1 > pair2
        if (a1 > a2 || (a1 == a2 && b1 > b2)) {
            // Swap both elements of pairs
            data[0] = a2; data[1] = b2;
            data[2] = a1; data[3] = b1;
        }
    }
}
```

---

## Alternative Solutions

### Using Struct-like Access

```cuda
__global__ void sortPairsKernel(int* pairs) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // Treat as pairs: pairs[0,1] = pair1, pairs[2,3] = pair2
        int* p1 = &pairs[0];  // p1[0] = a1, p1[1] = b1
        int* p2 = &pairs[2];  // p2[0] = a2, p2[1] = b2
        
        // Compare lexicographically
        if (p1[0] > p2[0] || (p1[0] == p2[0] && p1[1] > p2[1])) {
            // Swap pairs
            int temp0 = p1[0], temp1 = p1[1];
            p1[0] = p2[0]; p1[1] = p2[1];
            p2[0] = temp0; p2[1] = temp1;
        }
    }
}
```

### Device Function for Pair Comparison

```cuda
__device__ int comparePairs(int a1, int b1, int a2, int b2) {
    // Returns: negative if pair1 < pair2
    //          zero if pair1 == pair2
    //          positive if pair1 > pair2
    if (a1 != a2) return a1 - a2;
    return b1 - b2;
}

__global__ void sortPairsKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        if (comparePairs(data[0], data[1], data[2], data[3]) > 0) {
            int t0 = data[0], t1 = data[1];
            data[0] = data[2]; data[1] = data[3];
            data[2] = t0; data[3] = t1;
        }
    }
}
```

---

## Common Mistakes

### ❌ Only Comparing First Element
```cuda
if (a1 > a2) {
    // swap
}
// Wrong! Misses case where a1 == a2 but b1 > b2
```

### ❌ Wrong Comparison Order
```cuda
// Wrong: checking second element first
if (b1 > b2 || (b1 == b2 && a1 > a2)) {
    // This sorts by (b, a) not (a, b)!
}
```

### ❌ Swapping Only One Element
```cuda
if (shouldSwap) {
    // Wrong: only swapping 'a' values, not 'b'
    int temp = data[0];
    data[0] = data[2];
    data[2] = temp;
    // b values unchanged!
}
```

### ❌ Incorrect Operator in Comparison
```cuda
// This swaps when pair1 <= pair2 (wrong!)
if (a1 < a2 || (a1 == a2 && b1 < b2)) {
    // swap - puts larger first!
}
```

---

## Lexicographic Ordering Applications

| Application | Comparison |
|-------------|------------|
| Dictionary words | Letter by letter |
| Dates (YYYY-MM-DD) | Year, then month, then day |
| Version numbers | Major, minor, patch |
| 2D coordinates | X first, then Y |
| Student records | Name, then ID |

This problem introduces a fundamental comparison pattern used widely in sorting.

---

## Key Takeaways

1. **Lexicographic ordering** compares elements in sequence (primary, then secondary)
2. **Compound conditions** — first element dominates, second breaks ties
3. **Pair swapping** requires swapping both components together
4. **Memory layout** — pairs stored as consecutive array elements
5. **Foundation** for sorting complex data structures (tuples, records)

---

## Practice Exercises

1. Extend to sort **three pairs** lexicographically
2. Sort pairs in **descending** order
3. Handle pairs with **three elements** (a, b, c)
4. Implement parallel comparison for an array of N pairs

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/14)*
