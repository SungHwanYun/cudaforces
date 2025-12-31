# Problem ID Converter

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

In contests hosted on **cudaforces**, problem IDs are represented using uppercase letters according to the following rule:

- Problem 1 is A, Problem 2 is B, Problem 3 is C, ..., Problem 26 is Z

Given a problem number n, output the corresponding problem ID.

### Input
A single integer n is given on the first line.

**Constraints:**
- 1 ≤ n ≤ 26

### Output
Print the problem ID corresponding to problem n.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 1 | A |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 7 | G |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 26 | Z |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void convertKernel(int* n, char* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = 'A' + (*n - 1);
    }
}

int main() {
    int n;
    scanf("%d", &n);
    
    // Device memory
    int *d_n;
    char *d_result;
    cudaMalloc(&d_n, sizeof(int));
    cudaMalloc(&d_result, sizeof(char));
    
    // Copy input to device
    cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    convertKernel<<<1, 1>>>(d_n, d_result);
    
    // Copy result back to host
    char result;
    cudaMemcpy(&result, d_result, sizeof(char), cudaMemcpyDeviceToHost);
    
    printf("%c\n", result);
    
    cudaFree(d_n);
    cudaFree(d_result);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void convertKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs number-to-letter conversion on GPU |

---

## CUDA Concepts Covered

### 1. Character Arithmetic

Converting a 1-based index to a letter:

```cuda
*result = 'A' + (*n - 1);

// Examples:
// n = 1: 'A' + (1-1) = 'A' + 0 = 'A'
// n = 7: 'A' + (7-1) = 'A' + 6 = 'G'
// n = 26: 'A' + (26-1) = 'A' + 25 = 'Z'
```

### 2. ASCII Values

Characters are stored as ASCII values:

```
'A' = 65
'B' = 66
'C' = 67
...
'Z' = 90
```

So `'A' + 1 = 66 = 'B'`, etc.

### 3. Why n - 1?

The problem uses 1-based indexing, but character arithmetic is 0-based:

```
Problem 1 → 'A' (offset 0)
Problem 2 → 'B' (offset 1)
...
Problem n → 'A' + (n - 1)
```

### 4. Visualization

```
n:       1    2    3    4    5    6    7   ...   26
         │    │    │    │    │    │    │         │
n-1:     0    1    2    3    4    5    6   ...   25
         │    │    │    │    │    │    │         │
'A'+:   'A'  'B'  'C'  'D'  'E'  'F'  'G'  ...  'Z'
```

### 5. Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  n = 7                                                   │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (HostToDevice)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   d_n: [7]                                               │
│     │                                                    │
│     ▼                                                    │
│   'A' + (7 - 1) = 'A' + 6 = 'G'                          │
│     │                                                    │
│     ▼                                                    │
│   d_result: ['G']                                        │
│                                                          │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (DeviceToHost)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  result = 'G' → printf("G\n")                            │
└──────────────────────────────────────────────────────────┘
```

---

## Alternative Solutions

### Direct printf in Kernel

```cuda
__global__ void convertKernel(int* n) {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("%c\n", 'A' + (*n - 1));
    }
}
```

### Using Cast

```cuda
__global__ void convertKernel(int* n, char* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (char)('A' + *n - 1);
    }
}
```

### Using ASCII Value Directly

```cuda
__global__ void convertKernel(int* n, char* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // 'A' = 65 in ASCII
        *result = 64 + (*n);  // 64 + 1 = 65 = 'A'
    }
}
```

---

## Number-to-Letter Mapping

| n | n - 1 | 'A' + (n-1) | Result |
|---|-------|-------------|--------|
| 1 | 0 | 'A' + 0 | A |
| 2 | 1 | 'A' + 1 | B |
| 3 | 2 | 'A' + 2 | C |
| ... | ... | ... | ... |
| 7 | 6 | 'A' + 6 | G |
| ... | ... | ... | ... |
| 26 | 25 | 'A' + 25 | Z |

---

## Common Mistakes

### ❌ Forgetting to Subtract 1
```cuda
*result = 'A' + (*n);     // Wrong! n=1 gives 'B'
*result = 'A' + (*n - 1); // Correct - n=1 gives 'A'
```

### ❌ Using Lowercase
```cuda
*result = 'a' + (*n - 1); // Wrong! Gives lowercase
*result = 'A' + (*n - 1); // Correct - uppercase
```

### ❌ Wrong Format Specifier
```cuda
printf("%d\n", result);  // Wrong! Prints ASCII value (65)
printf("%c\n", result);  // Correct - prints character ('A')
```

### ❌ Off-by-One with 0-Based Input
```cuda
// If input were 0-based (0 to 25):
*result = 'A' + (*n);     // Would be correct
// But problem uses 1-based (1 to 26):
*result = 'A' + (*n - 1); // Correct for this problem
```

---

## Related Conversions

| From | To | Formula |
|------|-----|---------|
| Number (1-26) → Letter | `'A' + (n - 1)` | 1 → 'A' |
| Letter → Number (1-26) | `ch - 'A' + 1` | 'A' → 1 |
| Number (0-25) → Letter | `'A' + n` | 0 → 'A' |
| Letter → Number (0-25) | `ch - 'A'` | 'A' → 0 |

---

## Practical Application

This conversion is used in competitive programming:
- **Codeforces**: Problems are A, B, C, D, ...
- **AtCoder**: Same convention
- **ICPC**: Problems A through N typically
- **cudaforces**: Our platform!

```
Contest Problem List:
A. Hello World
B. Sum of Two
C. Maximum Element
D. Dynamic Programming
...
```

---

## Key Takeaways

1. **Character arithmetic** — characters can be added/subtracted
2. **ASCII values** — 'A' = 65, 'B' = 66, etc.
3. **1-based to 0-based** — subtract 1 for offset calculation
4. **%c format** — use for printing characters
5. **Common pattern** — widely used in competitive programming

---

## Practice Exercises

1. Convert **letter to number** (reverse: A → 1)
2. Handle **lowercase** letters (a-z)
3. Extend beyond 26 with **two-letter IDs** (AA, AB, ...)
4. Convert a **string of numbers** to letters

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/141)*
