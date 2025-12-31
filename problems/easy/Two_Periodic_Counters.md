# Two Periodic Counters

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

There is a time variable t (in seconds) and two integer counters x and y. Initially, at t = 0, the value of x is 3 and the value of y is 5.

The counter x increases by 3 every 2 seconds, and the counter y increases by 5 every 3 seconds.

In other words:
- x = 3, 3, 6, 6, 9, 9, 12, ... at t = 0, 1, 2, 3, 4, 5, 6, ...
- y = 5, 5, 5, 10, 10, 10, 15, ... at t = 0, 1, 2, 3, 4, 5, 6, ...

Given a time t, print the value of x + y at that moment.

### Input
A single line contains an integer t.

**Constraints:**
- 0 ≤ t ≤ 100

### Output
Print the value of x + y at time t.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 0 | 8 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 5 | 19 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 100 | 323 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void twoCountersKernel(int* t, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int time = *t;
        
        // x = initial + (t / period) * increment
        int x = 3 + (time / 2) * 3;
        
        // y = initial + (t / period) * increment
        int y = 5 + (time / 3) * 5;
        
        *result = x + y;
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
    twoCountersKernel<<<1, 1>>>(d_t, d_result);
    
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
| ✅ Kernel exists | `__global__ void twoCountersKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs dual counter calculation on GPU |

---

## CUDA Concepts Covered

### 1. Two Counter Formulas

Each counter follows the same pattern from the previous problem:

```cuda
// Counter x: initial=3, period=2, increment=3
int x = 3 + (time / 2) * 3;

// Counter y: initial=5, period=3, increment=5
int y = 5 + (time / 3) * 5;

// Result
*result = x + y;
```

### 2. Pattern Visualization

```
Time:   0   1   2   3   4   5   6   7   8   9   10
        │   │   │   │   │   │   │   │   │   │   │
x:      3───3   6───6   9───9  12──12  15──15  18
                ↑       ↑       ↑       ↑       ↑
              +3      +3      +3      +3      +3

y:      5───5───5  10──10──10  15──15──15  20──20
                ↑           ↑           ↑
              +5          +5          +5

x+y:    8   8  11  16  19  19  27  27  30  35  38
```

### 3. Verification Table

| t | t/2 | x = 3+(t/2)×3 | t/3 | y = 5+(t/3)×5 | x+y |
|---|-----|---------------|-----|---------------|-----|
| 0 | 0 | 3 | 0 | 5 | 8 |
| 1 | 0 | 3 | 0 | 5 | 8 |
| 2 | 1 | 6 | 0 | 5 | 11 |
| 3 | 1 | 6 | 1 | 10 | 16 |
| 4 | 2 | 9 | 1 | 10 | 19 |
| 5 | 2 | 9 | 1 | 10 | 19 |
| 6 | 3 | 12 | 2 | 15 | 27 |
| 100 | 50 | 153 | 33 | 170 | 323 |

### 4. Example Walkthrough

**Example 2**: `t = 5`
```
x = 3 + (5 / 2) × 3
  = 3 + 2 × 3
  = 3 + 6
  = 9

y = 5 + (5 / 3) × 5
  = 5 + 1 × 5
  = 5 + 5
  = 10

x + y = 9 + 10 = 19
```

**Example 3**: `t = 100`
```
x = 3 + (100 / 2) × 3
  = 3 + 50 × 3
  = 3 + 150
  = 153

y = 5 + (100 / 3) × 5
  = 5 + 33 × 5
  = 5 + 165
  = 170

x + y = 153 + 170 = 323
```

### 5. Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  Input: t = 5                                            │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (HostToDevice)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   d_t: [5]                                               │
│      │                                                   │
│   ┌──┴──────────────────────┐                            │
│   │                         │                            │
│   x = 3 + (5/2)×3       y = 5 + (5/3)×5                  │
│   x = 3 + 6 = 9         y = 5 + 5 = 10                   │
│   │                         │                            │
│   └────────────┬────────────┘                            │
│                │                                         │
│            x + y = 19                                    │
│                │                                         │
│                ▼                                         │
│         d_result: [19]                                   │
│                                                          │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (DeviceToHost)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  result = 19 → printf("19\n")                            │
└──────────────────────────────────────────────────────────┘
```

---

## Alternative Solutions

### Combined Formula

```cuda
__global__ void twoCountersKernel(int* t, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int time = *t;
        *result = 3 + (time / 2) * 3 + 5 + (time / 3) * 5;
    }
}
```

### Factored Form

```cuda
__global__ void twoCountersKernel(int* t, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int time = *t;
        // x = 3 * (1 + t/2), y = 5 * (1 + t/3)
        *result = 3 * (1 + time / 2) + 5 * (1 + time / 3);
    }
}
```

### Simulation with Loops

```cuda
__global__ void twoCountersKernel(int* t, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int time = *t;
        
        int x = 3;
        for (int i = 2; i <= time; i += 2) x += 3;
        
        int y = 5;
        for (int i = 3; i <= time; i += 3) y += 5;
        
        *result = x + y;
    }
}
```

---

## Counter Comparison

| Property | Counter x | Counter y |
|----------|-----------|-----------|
| Initial value | 3 | 5 |
| Period | 2 seconds | 3 seconds |
| Increment | +3 | +5 |
| Formula | 3 + (t/2)×3 | 5 + (t/3)×5 |
| At t=100 | 153 | 170 |

---

## Common Mistakes

### ❌ Mixing Up Periods
```cuda
int x = 3 + (time / 3) * 3;  // Wrong! x has period 2
int y = 5 + (time / 2) * 5;  // Wrong! y has period 3
```

### ❌ Mixing Up Increments
```cuda
int x = 3 + (time / 2) * 5;  // Wrong! x increments by 3
int y = 5 + (time / 3) * 3;  // Wrong! y increments by 5
```

### ❌ Forgetting Initial Values
```cuda
int x = (time / 2) * 3;      // Wrong! Missing initial 3
int y = (time / 3) * 5;      // Wrong! Missing initial 5
```

### ❌ Using Float Division
```cuda
int x = 3 + (time / 2.0) * 3;  // Wrong! Float division
int x = 3 + (time / 2) * 3;    // Correct - integer division
```

---

## General Pattern

For any periodic counter:

```cuda
value = initial + (time / period) * increment
```

| Parameter | Meaning |
|-----------|---------|
| initial | Starting value at t=0 |
| period | Time between updates |
| increment | Amount added each update |
| time | Current time |

This can be extended to N counters:
```cuda
total = Σ (initial_i + (t / period_i) × increment_i)
```

---

## Key Takeaways

1. **Combine independent counters** — calculate each separately, then sum
2. **Same formula pattern** — `initial + (t/period) × increment`
3. **Different periods** — each counter updates at its own rate
4. **Integer division** — key to step function behavior
5. **Extensible** — pattern works for any number of counters

---

## Practice Exercises

1. Add a **third counter** z with period 4 and increment 7
2. Find the **first time** when x + y exceeds a target value
3. Calculate **x × y** instead of x + y
4. Find when both counters **update simultaneously** (LCM of periods)

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/119)*
