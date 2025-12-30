# Prime Check

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Given an integer a, determine whether it is a prime number.

A prime number is an integer greater than 1 that is divisible only by 1 and itself.

### Input
A single integer a.

**Constraints:**
- 1 ≤ a ≤ 100

### Output
Print 1 if a is a prime number, otherwise print 0.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 7 | 1 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 10 | 0 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 1 | 0 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void primeCheckKernel(int* num, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int n = *num;
        
        // 1 is not prime
        if (n <= 1) {
            *result = 0;
            return;
        }
        
        // 2 is prime
        if (n == 2) {
            *result = 1;
            return;
        }
        
        // Even numbers > 2 are not prime
        if (n % 2 == 0) {
            *result = 0;
            return;
        }
        
        // Check odd divisors from 3 to sqrt(n)
        *result = 1;  // Assume prime
        for (int i = 3; i * i <= n; i += 2) {
            if (n % i == 0) {
                *result = 0;
                return;
            }
        }
    }
}

int main() {
    int a;
    scanf("%d", &a);
    
    // Device memory
    int *d_num, *d_result;
    cudaMalloc(&d_num, sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    // Copy input to device
    cudaMemcpy(d_num, &a, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    primeCheckKernel<<<1, 1>>>(d_num, d_result);
    
    // Copy result back to host
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d\n", result);
    
    cudaFree(d_num);
    cudaFree(d_result);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void primeCheckKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs prime number test with loop on GPU |

---

## CUDA Concepts Covered

### 1. Prime Number Definition

A prime number is:
- Greater than 1
- Divisible only by 1 and itself

```
Prime examples: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, ...
Not prime: 1 (by definition), 4 (2×2), 6 (2×3), 9 (3×3), 10 (2×5), ...
```

### 2. Trial Division Algorithm

Check if any number from 2 to √n divides n evenly:

```cuda
for (int i = 2; i * i <= n; i++) {
    if (n % i == 0) {
        // Found a divisor, not prime
        return 0;
    }
}
// No divisors found, is prime
return 1;
```

### 3. Why Check Only Up to √n?

If n = a × b, then at least one of a or b is ≤ √n.

```
Example: n = 100
  100 = 2 × 50
  100 = 4 × 25
  100 = 5 × 20
  100 = 10 × 10  ← √100 = 10
  
If we find no divisor ≤ 10, there's no divisor > 10 either.
```

### 4. Optimization: Skip Even Numbers

After checking 2, only check odd numbers:

```cuda
// Check 2 separately
if (n % 2 == 0) return n == 2;

// Only check odd divisors: 3, 5, 7, 9, ...
for (int i = 3; i * i <= n; i += 2) {
    if (n % i == 0) return 0;
}
```

### 5. Example Walkthrough

**Example 1**: `n = 7`
```
n > 1? Yes
n == 2? No
n % 2 == 0? No (7 is odd)

Check divisors: i = 3
  3 * 3 = 9 > 7? Yes → Loop ends

No divisor found → Prime (1)
```

**Example 2**: `n = 10`
```
n > 1? Yes
n == 2? No
n % 2 == 0? Yes (10 is even) → Not Prime (0)
```

**Example 3**: `n = 1`
```
n > 1? No → Not Prime (0)
```

---

## Alternative Solutions

### Simple Loop (No Optimizations)

```cuda
__global__ void primeCheckKernel(int* num, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int n = *num;
        
        if (n <= 1) {
            *result = 0;
            return;
        }
        
        *result = 1;
        for (int i = 2; i < n; i++) {
            if (n % i == 0) {
                *result = 0;
                return;
            }
        }
    }
}
```

### Using sqrt Function

```cuda
__global__ void primeCheckKernel(int* num, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int n = *num;
        
        if (n <= 1) {
            *result = 0;
            return;
        }
        
        int sqrtN = (int)sqrtf((float)n);
        *result = 1;
        
        for (int i = 2; i <= sqrtN; i++) {
            if (n % i == 0) {
                *result = 0;
                return;
            }
        }
    }
}
```

### Hardcoded for Small Range (1-100)

```cuda
__global__ void primeCheckKernel(int* num, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int n = *num;
        
        // Primes up to 100
        int primes[] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,
                        53,59,61,67,71,73,79,83,89,97};
        int count = 25;
        
        *result = 0;
        for (int i = 0; i < count; i++) {
            if (primes[i] == n) {
                *result = 1;
                return;
            }
        }
    }
}
```

---

## Prime Numbers 1-100

```
 2   3   5   7  11  13  17  19  23  29
31  37  41  43  47  53  59  61  67  71
73  79  83  89  97

Total: 25 prime numbers between 1 and 100
```

### Visualization

```
    1   2   3   4   5   6   7   8   9  10
   11  12  13  14  15  16  17  18  19  20
   21  22  23  24  25  26  27  28  29  30
   31  32  33  34  35  36  37  38  39  40
   41  42  43  44  45  46  47  48  49  50
   51  52  53  54  55  56  57  58  59  60
   61  62  63  64  65  66  67  68  69  70
   71  72  73  74  75  76  77  78  79  80
   81  82  83  84  85  86  87  88  89  90
   91  92  93  94  95  96  97  98  99 100

Circled numbers are prime.
```

---

## Common Mistakes

### ❌ Treating 1 as Prime
```cuda
if (n < 1) return 0;  // Wrong! 1 should also return 0
if (n <= 1) return 0; // Correct
```

### ❌ Not Handling 2 Correctly
```cuda
// If starting loop at 2, need to handle n=2 case
if (n == 2) return 1;  // 2 is the only even prime
```

### ❌ Checking Too Many Numbers
```cuda
for (int i = 2; i <= n; i++)  // Wrong! Includes n itself
for (int i = 2; i < n; i++)   // OK but inefficient
for (int i = 2; i * i <= n; i++) // Best - only up to sqrt(n)
```

### ❌ Integer Overflow in i*i
```cuda
// For very large n, i*i might overflow
// Solution: use i <= n/i or compare with sqrt
```

---

## Algorithm Complexity

| Algorithm | Time Complexity | For n = 100 |
|-----------|----------------|-------------|
| Check all 2 to n-1 | O(n) | 98 checks |
| Check 2 to √n | O(√n) | 10 checks |
| Skip evens | O(√n / 2) | 5 checks |

---

## Loop in CUDA Kernels

This problem introduces loops inside kernels:

```cuda
for (int i = 3; i * i <= n; i += 2) {
    if (n % i == 0) {
        *result = 0;
        return;  // Early exit
    }
}
```

Key points:
- Loops work the same as in C
- `return` exits the kernel function
- Avoid infinite loops (GPU will hang!)

---

## Key Takeaways

1. **1 is not prime** — must be greater than 1
2. **2 is the only even prime** — handle separately
3. **Check up to √n** — optimization for efficiency
4. **Early return** — stop as soon as divisor is found
5. **Loops in kernels** — standard C loops work in CUDA

---

## Practice Exercises

1. Find all **prime numbers** between 1 and N
2. Check if a number is a **perfect square**
3. Find the **smallest prime factor** of a number
4. Count the **number of divisors** of N

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/104)*
