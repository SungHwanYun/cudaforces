# Multiply Three Integers

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Given three integers a, b, and c, compute and print their product a × b × c.

### Input
A single line containing three integers a, b, and c, separated by spaces.

**Constraints:**
- 1 ≤ a, b, c ≤ 1,000

### Output
Print the value of a × b × c.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 1 2 3 | 6 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 10 20 30 | 6000 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 100 100 100 | 1000000 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void multiplyKernel(int* a, int* b, int* c, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*a) * (*b) * (*c);
    }
}

int main() {
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);
    
    // Device memory
    int *d_a, *d_b, *d_c, *d_result;
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    cudaMalloc(&d_c, sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    // Copy inputs to device
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &c, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    multiplyKernel<<<1, 1>>>(d_a, d_b, d_c, d_result);
    
    // Copy result back to host
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d\n", result);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_result);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void multiplyKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs triple multiplication on GPU |

---

## CUDA Concepts Covered

### 1. Triple Multiplication

Multiply three integers together:

```cuda
*result = (*a) * (*b) * (*c);

// Examples:
// 1 × 2 × 3 = 6
// 10 × 20 × 30 = 6000
// 100 × 100 × 100 = 1,000,000
```

### 2. Associativity of Multiplication

Multiplication is associative — order of operations doesn't matter:

```cuda
// All equivalent:
*result = (*a) * (*b) * (*c);
*result = (*a) * ((*b) * (*c));
*result = ((*a) * (*b)) * (*c);
```

### 3. Visualization

```
Example: a = 10, b = 20, c = 30

Step 1:  10 × 20 = 200
Step 2: 200 × 30 = 6000

   10      20      30
    │       │       │
    └───┬───┘       │
        │           │
       200          │
        │           │
        └─────┬─────┘
              │
            6000
```

### 4. Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  a = 10, b = 20, c = 30                                  │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (HostToDevice)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   d_a: [10]    d_b: [20]    d_c: [30]                    │
│         \          │          /                          │
│          \         │         /                           │
│           \        │        /                            │
│            ───────××───────                              │
│                   │                                      │
│                   ▼                                      │
│            d_result: [6000]                              │
│                                                          │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (DeviceToHost)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  result = 6000 → printf("6000\n")                        │
└──────────────────────────────────────────────────────────┘
```

### 5. Result Range

With constraints 1 ≤ a, b, c ≤ 1,000:
- Minimum: 1 × 1 × 1 = 1
- Maximum: 1000 × 1000 × 1000 = 1,000,000,000 (1 billion)

This fits within int range (2³¹ - 1 ≈ 2.1 billion).

---

## Alternative Solutions

### Using Array

```cuda
__global__ void multiplyKernel(int* data, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = data[0] * data[1] * data[2];
    }
}

int main() {
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);
    
    int h_data[3] = {a, b, c};
    int h_result;
    
    int *d_data, *d_result;
    cudaMalloc(&d_data, 3 * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    cudaMemcpy(d_data, h_data, 3 * sizeof(int), cudaMemcpyHostToDevice);
    
    multiplyKernel<<<1, 1>>>(d_data, d_result);
    
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d\n", h_result);
    
    cudaFree(d_data);
    cudaFree(d_result);
    
    return 0;
}
```

### Direct Output in Kernel

```cuda
__global__ void multiplyKernel(int* a, int* b, int* c) {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("%d\n", (*a) * (*b) * (*c));
    }
}
```

### Step-by-Step Multiplication

```cuda
__global__ void multiplyKernel(int* data, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int temp = data[0] * data[1];  // First two
        *result = temp * data[2];       // Then third
    }
}
```

---

## Arithmetic Operations Progression

| Problem | Operation | Inputs | Formula |
|---------|-----------|--------|---------|
| Sum of Two | Addition | 2 | a + b |
| Difference | Subtraction | 2 | a - b |
| Product of Two | Multiplication | 2 | a × b |
| **Product of Three** | Multiplication | 3 | a × b × c |
| Division | Division | 2 | a / b |
| Remainder | Modulo | 2 | a % b |

---

## Common Mistakes

### ❌ Missing Parentheses with Pointers
```cuda
*result = *a * *b * *c;      // Works but confusing
*result = (*a) * (*b) * (*c); // Clearer
```

### ❌ Integer Overflow (Not in This Problem)
```cuda
// If values were larger (e.g., up to 10^6):
// 10^6 × 10^6 × 10^6 = 10^18 → Overflow!
// Would need long long
long long result = (long long)a * b * c;
```

### ❌ Wrong Format Specifier
```cuda
printf("%d\n", result);   // Correct for int
printf("%lld\n", result); // Would be needed for long long
```

---

## Multiplication Properties

| Property | Description | Example |
|----------|-------------|---------|
| Commutative | a × b = b × a | 2 × 3 = 3 × 2 = 6 |
| Associative | (a × b) × c = a × (b × c) | (2 × 3) × 4 = 2 × (3 × 4) = 24 |
| Identity | a × 1 = a | 5 × 1 = 5 |
| Zero | a × 0 = 0 | 5 × 0 = 0 |

---

## Key Takeaways

1. **Triple multiplication** — extension of binary multiplication
2. **Associativity** — order of operations doesn't affect result
3. **Result range** — check for potential overflow
4. **Pointer dereferencing** — use parentheses for clarity
5. **Commutative** — a × b × c = c × b × a

---

## Practice Exercises

1. Multiply **four or more** integers
2. Calculate the **average** of three numbers (sum then divide)
3. Compute **power**: a^b using repeated multiplication
4. Find the **volume** of a box (length × width × height)

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/138)*
