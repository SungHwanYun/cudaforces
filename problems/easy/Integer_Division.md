# Integer Division

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given two integers a and b. Print the quotient a ÷ b (integer division).

### Input
The first line contains an integer a.

The second line contains an integer b.

**Constraints:**
- 1 ≤ a, b ≤ 100
- b ≠ 0

### Output
Print the quotient a ÷ b (integer division) on the first line.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 13<br>5 | 2 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 100<br>10 | 10 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 7<br>8 | 0 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void divideKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*a) / (*b);
    }
}

int main() {
    int a, b;
    scanf("%d", &a);
    scanf("%d", &b);
    
    // Device memory
    int *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    // Copy inputs to device
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    divideKernel<<<1, 1>>>(d_a, d_b, d_result);
    
    // Copy result back to host
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d\n", result);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void divideKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs integer division on GPU |

---

## CUDA Concepts Covered

### 1. Integer Division

In C/CUDA, dividing two integers produces an integer (truncated toward zero):

```cuda
*result = (*a) / (*b);

// Examples:
// 13 / 5 = 2   (not 2.6)
// 7 / 8 = 0    (not 0.875)
// 100 / 10 = 10
```

The fractional part is **discarded**, not rounded.

### 2. Truncation Toward Zero

```
13 ÷ 5 = 2.6
       ↓
Integer division: 2 (truncated)

7 ÷ 8 = 0.875
      ↓
Integer division: 0 (truncated)
```

### 3. Order Matters (Non-Commutative)

Unlike addition and multiplication, division is **not commutative**:

```cuda
*result = (*a) / (*b);  // a ÷ b
*result = (*b) / (*a);  // b ÷ a (different result!)

// Example: a=13, b=5
// a / b = 13 / 5 = 2
// b / a = 5 / 13 = 0
```

### 4. Data Flow Visualization

```
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  a = 13, b = 5                                           │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (HostToDevice)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   d_a: [13]    d_b: [5]                                  │
│          \      /                                        │
│           \    /                                         │
│            ÷   ← Division: 13 ÷ 5 = 2 (truncated)        │
│            │                                             │
│            ▼                                             │
│   d_result: [2]                                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (DeviceToHost)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  result = 2 → printf("2\n")                              │
└──────────────────────────────────────────────────────────┘
```

### 5. Zero Result Case

When divisor > dividend, result is 0:

```
7 / 8 = 0  (since 7 < 8)
1 / 100 = 0
99 / 100 = 0
```

---

## Alternative Solutions

### Using Array

```cuda
__global__ void divideKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        data[2] = data[0] / data[1];
    }
}

int main() {
    int a, b;
    scanf("%d", &a);
    scanf("%d", &b);
    
    int h_data[3] = {a, b, 0};
    
    int* d_data;
    cudaMalloc(&d_data, 3 * sizeof(int));
    cudaMemcpy(d_data, h_data, 3 * sizeof(int), cudaMemcpyHostToDevice);
    
    divideKernel<<<1, 1>>>(d_data);
    
    cudaMemcpy(h_data, d_data, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d\n", h_data[2]);
    
    cudaFree(d_data);
    
    return 0;
}
```

### Direct Output in Kernel

```cuda
__global__ void divideKernel(int* a, int* b) {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("%d\n", (*a) / (*b));
    }
}
```

---

## Integer Division vs Float Division

| Aspect | Integer Division | Float Division |
|--------|------------------|----------------|
| Operator | `/` (int operands) | `/` (float operands) |
| Result type | `int` | `float` |
| `13 / 5` | `2` | `2.6f` |
| `7 / 8` | `0` | `0.875f` |
| Truncation | Yes (toward zero) | No |

To get float division:

```cuda
float result = (float)a / b;  // Cast at least one operand to float
```

---

## Comparison: All Four Basic Operations

| Operation | Symbol | Commutative | `13 op 5` | `5 op 13` |
|-----------|--------|-------------|-----------|-----------|
| Addition | `+` | Yes | 18 | 18 |
| Subtraction | `-` | No | 8 | -8 |
| Multiplication | `*` | Yes | 65 | 65 |
| **Division** | `/` | No | 2 | 0 |

---

## Common Mistakes

### ❌ Division by Zero
```cuda
// If b could be 0 (not in this problem, but be aware):
*result = (*a) / (*b);  // Runtime error if *b == 0!
```
Always validate inputs in production code.

### ❌ Wrong Operand Order
```cuda
*result = (*b) / (*a);  // Wrong! Should be (*a) / (*b)
```
Order matters for division.

### ❌ Expecting Float Result
```cuda
int a = 13, b = 5;
int result = a / b;     // result = 2, not 2.6
printf("%d\n", result); // Prints: 2
```

### ❌ Confusing `/` and `%`
```cuda
*result = (*a) % (*b);  // Wrong! This is modulo, not division
*result = (*a) / (*b);  // Correct - division
```

---

## Division in CUDA Index Calculations

Division is commonly used in parallel computing:

```cuda
// Calculate which block a global index belongs to
int blockIndex = globalIdx / blockDim.x;

// Calculate how many blocks needed for N elements
int numBlocks = (N + blockSize - 1) / blockSize;  // Ceiling division
```

### Ceiling Division Pattern

To divide and round **up** instead of down:

```cuda
// Integer ceiling division: ceil(a / b)
int ceilDiv = (a + b - 1) / b;

// Example: 13 elements, block size 5
// (13 + 5 - 1) / 5 = 17 / 5 = 3 blocks (not 2!)
```

---

## Key Takeaways

1. **Integer division truncates** — fractional part discarded
2. **Not commutative** — a/b ≠ b/a
3. **Zero result** when dividend < divisor
4. **No division by zero** — always validate if divisor could be 0
5. **Foundation for indexing** — used in block/thread calculations

---

## Practice Exercises

1. Compute both **quotient and remainder** (a/b and a%b)
2. Implement **ceiling division** (round up instead of truncate)
3. Check if a is **divisible** by b (remainder == 0)
4. Compute average of two integers: (a + b) / 2

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/24)*
