# Remainder Operation

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Given two integers a and b, compute a % b. The remainder operation returns the remainder when a is divided by b. For example, 7 % 3 = 1.

### Input
The first line contains two integers a and b separated by a space (1 ≤ a, b ≤ 100).

### Output
Print a % b on the first line.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 2 1 | 0 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 7 3 | 1 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 15 4 | 3 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void remainderKernel(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*a) % (*b);
    }
}

int main() {
    int a, b;
    scanf("%d %d", &a, &b);
    
    // Device memory
    int *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    // Copy inputs to device
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    remainderKernel<<<1, 1>>>(d_a, d_b, d_result);
    
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
| ✅ Kernel exists | `__global__ void remainderKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs modulo operation on GPU |

---

## CUDA Concepts Covered

### 1. Modulo Operator

The `%` operator returns the remainder after integer division:

```cuda
*result = (*a) % (*b);

// Examples:
// 7 % 3 = 1   (7 = 3*2 + 1)
// 15 % 4 = 3  (15 = 4*3 + 3)
// 2 % 1 = 0   (2 = 1*2 + 0)
```

### 2. Relationship: Division and Modulo

Division (`/`) and modulo (`%`) are complementary:

```
a = b * (a / b) + (a % b)

Example: a = 15, b = 4
  15 / 4 = 3  (quotient)
  15 % 4 = 3  (remainder)
  Verify: 4 * 3 + 3 = 15 ✓
```

### 3. Modulo Properties

```
Property 1: 0 ≤ (a % b) < b  (for positive a, b)

Property 2: a % b = 0 when a is divisible by b

Property 3: a % 1 = 0 (everything is divisible by 1)

Property 4: a % a = 0 (any number mod itself is 0)

Property 5: a % b = a when a < b
```

### 4. Data Flow Visualization

```
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  a = 7, b = 3                                            │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (HostToDevice)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   d_a: [7]     d_b: [3]                                  │
│          \      /                                        │
│           \    /                                         │
│            %   ← Modulo: 7 % 3 = 1                       │
│            │                                             │
│            ▼                                             │
│   d_result: [1]                                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (DeviceToHost)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  result = 1 → printf("1\n")                              │
└──────────────────────────────────────────────────────────┘
```

### 5. Visual Explanation

```
Example: 15 % 4 = ?

15 ÷ 4 = 3 remainder 3

   ┌───┬───┬───┬───┐ ┌───┬───┬───┬───┐ ┌───┬───┬───┬───┐ ┌───┬───┬───┐
   │ 1 │ 2 │ 3 │ 4 │ │ 5 │ 6 │ 7 │ 8 │ │ 9 │10 │11 │12 │ │13 │14 │15 │
   └───┴───┴───┴───┘ └───┴───┴───┴───┘ └───┴───┴───┴───┘ └───┴───┴───┘
      Group 1            Group 2            Group 3        Remainder
        (4)                (4)                (4)             (3)

15 = 4 × 3 + 3, so 15 % 4 = 3
```

---

## Alternative Solutions

### Using Array

```cuda
__global__ void remainderKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        data[2] = data[0] % data[1];
    }
}

int main() {
    int a, b;
    scanf("%d %d", &a, &b);
    
    int h_data[3] = {a, b, 0};
    
    int* d_data;
    cudaMalloc(&d_data, 3 * sizeof(int));
    cudaMemcpy(d_data, h_data, 3 * sizeof(int), cudaMemcpyHostToDevice);
    
    remainderKernel<<<1, 1>>>(d_data);
    
    cudaMemcpy(h_data, d_data, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d\n", h_data[2]);
    
    cudaFree(d_data);
    
    return 0;
}
```

### Direct Output in Kernel

```cuda
__global__ void remainderKernel(int* a, int* b) {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("%d\n", (*a) % (*b));
    }
}
```

---

## Division vs Modulo Comparison

| Operation | Symbol | `15 op 4` | `7 op 3` | `2 op 1` |
|-----------|--------|-----------|----------|----------|
| Division | `/` | 3 | 2 | 2 |
| **Modulo** | `%` | 3 | 1 | 0 |

Together they satisfy: `a = b * (a/b) + (a%b)`

---

## Common Mistakes

### ❌ Confusing Division and Modulo
```cuda
*result = (*a) / (*b);  // Wrong! This is division
*result = (*a) % (*b);  // Correct - modulo
```

### ❌ Modulo by Zero
```cuda
// If b could be 0 (not in this problem):
*result = (*a) % (*b);  // Runtime error if *b == 0!
```

### ❌ Wrong Operand Order
```cuda
*result = (*b) % (*a);  // Wrong! Should be (*a) % (*b)
```
Order matters: a % b ≠ b % a (usually).

### ❌ Expecting Float Result
```cuda
// Modulo always returns an integer
// There's no fractional part
7 % 3 = 1  // Not 1.something
```

---

## Modulo Use Cases in CUDA

| Use Case | Example | Purpose |
|----------|---------|---------|
| Wrap-around | `idx % arraySize` | Circular buffer access |
| Even/Odd check | `n % 2` | 0 = even, 1 = odd |
| Alphabet wrap | `(pos + shift) % 26` | Caesar cipher |
| Grid cycling | `threadIdx.x % 4` | Distribute work |
| Hash function | `key % tableSize` | Array indexing |

### Example: Thread Distribution

```cuda
__global__ void distributedWork(int* data, int n) {
    int idx = threadIdx.x;
    int workType = idx % 4;  // 0, 1, 2, 3, 0, 1, 2, 3, ...
    
    // Different threads handle different types of work
    if (workType == 0) { /* Type A */ }
    else if (workType == 1) { /* Type B */ }
    // ...
}
```

---

## Complete Arithmetic Operations Summary

| Operation | Symbol | Commutative | Example |
|-----------|--------|-------------|---------|
| Addition | `+` | Yes | 3 + 5 = 8 |
| Subtraction | `-` | No | 5 - 3 = 2 |
| Multiplication | `*` | Yes | 3 × 5 = 15 |
| Division | `/` | No | 7 / 3 = 2 |
| **Modulo** | `%` | No | 7 % 3 = 1 |

---

## Key Takeaways

1. **Modulo returns remainder** after integer division
2. **Result range**: 0 ≤ result < b (for positive inputs)
3. **Not commutative**: a % b ≠ b % a
4. **Zero result** when a is divisible by b
5. **Essential for cycling** — wrap-around, even/odd, circular access

---

## Practice Exercises

1. Compute **both quotient and remainder** in one kernel
2. Check if a number is **divisible by 3** using modulo
3. Implement **circular array access** using modulo
4. Find the **last digit** of a number (n % 10)

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/32)*
