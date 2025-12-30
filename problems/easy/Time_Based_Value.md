# Time-Based Value

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

There is a time variable t and an integer variable x. At time t = 0, the value of x is 3.

The value of x increases by 3 every 2 seconds. That is, x = 3 at t = 0, x = 3 at t = 1, x = 6 at t = 2, x = 6 at t = 3, x = 9 at t = 4, and so on.

Given a time t, output the value of x at time t.

### Input
A single line contains an integer t.

**Constraints:**
- 0 ≤ t ≤ 100

### Output
Output a single integer: the value of x at time t.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 0 | 3 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 5 | 9 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 10 | 18 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void timeValueKernel(int* t, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // x = initial_value + (number_of_increases) * increment
        // x = 3 + (t/2) * 3
        *result = 3 + (*t / 2) * 3;
    }
}

int main() {
    int t;
    scanf("%d", &t);
    
    // Device memory
    int *d_t, *d_result;
    cudaMalloc(&d_t, sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    // Copy input to device
    cudaMemcpy(d_t, &t, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    timeValueKernel<<<1, 1>>>(d_t, d_result);
    
    // Copy result back to host
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d\n", result);
    
    cudaFree(d_t);
    cudaFree(d_result);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void timeValueKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs time-based calculation on GPU |

---

## CUDA Concepts Covered

### 1. Pattern Analysis

First, let's understand the pattern:

```
t:  0   1   2   3   4   5   6   7   8   9   10
x:  3   3   6   6   9   9  12  12  15  15  18

Observation:
- x starts at 3
- x increases by 3 at t = 2, 4, 6, 8, 10, ...
- x stays the same for 2 consecutive time units
```

### 2. Formula Derivation

```
Number of increases = t / 2 (integer division)

x = initial_value + (number_of_increases) × increment
x = 3 + (t / 2) × 3
```

**Verification:**
| t | t/2 | 3 + (t/2)×3 | Expected |
|---|-----|-------------|----------|
| 0 | 0 | 3 + 0 = 3 | 3 ✓ |
| 1 | 0 | 3 + 0 = 3 | 3 ✓ |
| 2 | 1 | 3 + 3 = 6 | 6 ✓ |
| 5 | 2 | 3 + 6 = 9 | 9 ✓ |
| 10 | 5 | 3 + 15 = 18 | 18 ✓ |

### 3. Visualization

```
Time:    0   1   2   3   4   5   6   7   8   9   10
         │   │   │   │   │   │   │   │   │   │   │
Value:   3───3   6───6   9───9  12──12  15──15  18
              ↑       ↑       ↑       ↑       ↑
            +3      +3      +3      +3      +3
         (at t=2) (at t=4) (at t=6) (at t=8) (at t=10)
```

### 4. Alternative Formula

The formula can also be written as:

```cuda
*result = 3 * (1 + *t / 2);

// Or factored:
*result = 3 * ((*t / 2) + 1);
```

Both are equivalent:
- `3 + (t/2) * 3`
- `3 * (1 + t/2)`

### 5. Integer Division Role

```cuda
// Integer division truncates toward zero
5 / 2 = 2   (not 2.5)
1 / 2 = 0   (not 0.5)

// This naturally handles the "stay same for 2 seconds" behavior
t = 0: 0/2 = 0 increases
t = 1: 1/2 = 0 increases (same as t=0)
t = 2: 2/2 = 1 increase
t = 3: 3/2 = 1 increase (same as t=2)
```

---

## Alternative Solutions

### Using Factored Formula

```cuda
__global__ void timeValueKernel(int* t, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = 3 * (1 + *t / 2);
    }
}
```

### Using Loop (Simulation)

```cuda
__global__ void timeValueKernel(int* t, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int x = 3;
        for (int i = 2; i <= *t; i += 2) {
            x += 3;
        }
        *result = x;
    }
}
```

### Using Conditional for Clarity

```cuda
__global__ void timeValueKernel(int* t, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int time = *t;
        int increments = time / 2;  // How many 2-second intervals passed
        *result = 3 + increments * 3;
    }
}
```

---

## Time-Value Table

| t | t/2 | x |
|---|-----|---|
| 0 | 0 | 3 |
| 1 | 0 | 3 |
| 2 | 1 | 6 |
| 3 | 1 | 6 |
| 4 | 2 | 9 |
| 5 | 2 | 9 |
| 6 | 3 | 12 |
| 7 | 3 | 12 |
| 8 | 4 | 15 |
| 9 | 4 | 15 |
| 10 | 5 | 18 |

---

## Common Mistakes

### ❌ Using Float Division
```cuda
*result = 3 + (t / 2.0) * 3;  // Wrong! Float division gives different results
*result = 3 + (t / 2) * 3;    // Correct - integer division
```

### ❌ Off-by-One in Starting Value
```cuda
*result = (t / 2) * 3;        // Wrong! Missing initial value 3
*result = 3 + (t / 2) * 3;    // Correct
```

### ❌ Wrong Period
```cuda
*result = 3 + (t / 3) * 3;    // Wrong! Period is 2, not 3
*result = 3 + (t / 2) * 3;    // Correct
```

### ❌ Wrong Increment
```cuda
*result = 3 + (t / 2) * 2;    // Wrong! Increment is 3, not 2
*result = 3 + (t / 2) * 3;    // Correct
```

---

## Discrete Time Patterns

This problem demonstrates a **step function** (staircase pattern):

```
x
18│                                              ▄▄▄
15│                                      ▄▄▄▄▄▄▄
12│                              ▄▄▄▄▄▄▄
 9│                      ▄▄▄▄▄▄▄
 6│              ▄▄▄▄▄▄▄
 3│      ▄▄▄▄▄▄▄
  └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──→ t
     0  1  2  3  4  5  6  7  8  9  10
```

---

## General Formula Pattern

For similar problems with different parameters:

```
x(t) = initial + (t / period) × increment

Where:
- initial = starting value (3 in this problem)
- period = time between increases (2 in this problem)
- increment = amount of each increase (3 in this problem)
```

---

## Key Takeaways

1. **Integer division** naturally handles discrete time intervals
2. **Pattern recognition** — identify period, increment, and initial value
3. **Formula approach** — more efficient than simulation
4. **Step function** — value stays constant between updates
5. **Factoring** — `3 + (t/2) × 3` = `3 × (1 + t/2)`

---

## Practice Exercises

1. Modify to start at x = 5 with increment of 2 every 3 seconds
2. Calculate x at time t with **continuous** change (linear interpolation)
3. Find the **first time** when x reaches a target value
4. Handle **negative time** (value decreases going backward)

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/117)*
