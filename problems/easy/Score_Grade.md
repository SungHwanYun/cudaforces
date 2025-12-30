# Score Grade

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

A student has two test scores a and b. Determine the final grade based on the following criteria:

- **A**: Both scores are 90 or above (a ≥ 90 and b ≥ 90)
- **B**: Only the first score is 90 or above (a ≥ 90 and b < 90)
- **C**: Only the second score is 90 or above (a < 90 and b ≥ 90)
- **D**: Both scores are below 90 (a < 90 and b < 90)

### Input
A single line containing two space-separated integers a and b.

**Constraints:**
- 80 ≤ a, b ≤ 100

### Output
Print the corresponding grade: A, B, C, or D.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 90 90 | A |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 95 85 | B |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 88 92 | C |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void gradeKernel(int* a, int* b, char* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int scoreA = *a;
        int scoreB = *b;
        
        if (scoreA >= 90 && scoreB >= 90) {
            *result = 'A';
        } else if (scoreA >= 90 && scoreB < 90) {
            *result = 'B';
        } else if (scoreA < 90 && scoreB >= 90) {
            *result = 'C';
        } else {
            *result = 'D';
        }
    }
}

int main() {
    int a, b;
    scanf("%d %d", &a, &b);
    
    // Device memory
    int *d_a, *d_b;
    char *d_result;
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    cudaMalloc(&d_result, sizeof(char));
    
    // Copy inputs to device
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    gradeKernel<<<1, 1>>>(d_a, d_b, d_result);
    
    // Copy result back to host
    char result;
    cudaMemcpy(&result, d_result, sizeof(char), cudaMemcpyDeviceToHost);
    
    printf("%c\n", result);
    
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
| ✅ Kernel exists | `__global__ void gradeKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs conditional logic on GPU |

---

## CUDA Concepts Covered

### 1. Logical AND Operator

The `&&` operator combines two conditions:

```cuda
if (scoreA >= 90 && scoreB >= 90) {
    *result = 'A';  // Both conditions must be true
}
```

### 2. Decision Tree Visualization

```
                    ┌─────────────────┐
                    │  a ≥ 90 ?       │
                    └────────┬────────┘
                    YES      │      NO
              ┌──────────────┴──────────────┐
              ▼                             ▼
     ┌─────────────────┐           ┌─────────────────┐
     │   b ≥ 90 ?      │           │   b ≥ 90 ?      │
     └────────┬────────┘           └────────┬────────┘
     YES      │      NO            YES      │      NO
     ▼        ▼                    ▼        ▼
    'A'      'B'                  'C'      'D'
```

### 3. Four Cases Matrix

```
          │  b ≥ 90  │  b < 90
──────────┼──────────┼──────────
 a ≥ 90   │    A     │    B
──────────┼──────────┼──────────
 a < 90   │    C     │    D
```

### 4. If-Else Chain

```cuda
if (scoreA >= 90 && scoreB >= 90) {
    *result = 'A';       // Both high
} else if (scoreA >= 90 && scoreB < 90) {
    *result = 'B';       // Only first high
} else if (scoreA < 90 && scoreB >= 90) {
    *result = 'C';       // Only second high
} else {
    *result = 'D';       // Both low
}
```

### 5. Example Walkthrough

**Example 1**: `a = 90, b = 90`
```
Check: 90 >= 90 && 90 >= 90
       true   &&   true
       = true → 'A'
```

**Example 2**: `a = 95, b = 85`
```
Check: 95 >= 90 && 85 >= 90
       true   &&  false
       = false → next condition

Check: 95 >= 90 && 85 < 90
       true   &&  true
       = true → 'B'
```

**Example 3**: `a = 88, b = 92`
```
Check: 88 >= 90 && 92 >= 90
       false  &&  true
       = false → next condition

Check: 88 >= 90 && 92 < 90
       false  && false
       = false → next condition

Check: 88 < 90 && 92 >= 90
       true  &&  true
       = true → 'C'
```

---

## Alternative Solutions

### Nested If-Else (More Efficient)

```cuda
__global__ void gradeKernel(int* a, int* b, char* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        if (*a >= 90) {
            if (*b >= 90) {
                *result = 'A';
            } else {
                *result = 'B';
            }
        } else {
            if (*b >= 90) {
                *result = 'C';
            } else {
                *result = 'D';
            }
        }
    }
}
```

### Using Boolean Variables

```cuda
__global__ void gradeKernel(int* a, int* b, char* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int highA = (*a >= 90);  // 1 if true, 0 if false
        int highB = (*b >= 90);
        
        if (highA && highB) *result = 'A';
        else if (highA) *result = 'B';
        else if (highB) *result = 'C';
        else *result = 'D';
    }
}
```

### Using Arithmetic Index

```cuda
__global__ void gradeKernel(int* a, int* b, char* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // Create 2-bit index: (a>=90)*2 + (b>=90)
        // Index: 0=D, 1=C, 2=B, 3=A
        char grades[] = {'D', 'C', 'B', 'A'};
        int index = ((*a >= 90) << 1) | (*b >= 90);
        *result = grades[index];
    }
}
```

---

## Logical Operators Reference

| Operator | Name | Usage | Result |
|----------|------|-------|--------|
| `&&` | AND | `a && b` | true if both true |
| `\|\|` | OR | `a \|\| b` | true if either true |
| `!` | NOT | `!a` | true if a is false |

### AND Truth Table

| a | b | a && b |
|---|---|--------|
| T | T | T |
| T | F | F |
| F | T | F |
| F | F | F |

---

## Common Mistakes

### ❌ Using Single & Instead of &&
```cuda
if (scoreA >= 90 & scoreB >= 90)  // Bitwise AND (different!)
if (scoreA >= 90 && scoreB >= 90) // Correct: Logical AND
```

### ❌ Missing Conditions
```cuda
// Wrong: Doesn't cover all cases properly
if (scoreA >= 90) *result = 'A';  // What about B?
if (scoreB >= 90) *result = 'C';  // Overwrites A!
```

### ❌ Wrong Comparison Direction
```cuda
if (scoreA > 90 && scoreB > 90)   // Wrong! Excludes exactly 90
if (scoreA >= 90 && scoreB >= 90) // Correct: includes 90
```

### ❌ Mutually Exclusive Conditions Issue
```cuda
// Wrong order can cause issues
if (scoreA >= 90) *result = 'B';     // Always assigns B for high A
else if (scoreA >= 90 && scoreB >= 90) // Never reached!
```

---

## Boolean Logic Patterns

```
A: a ≥ 90 AND b ≥ 90    (both conditions)
B: a ≥ 90 AND b < 90    (first only)
C: a < 90 AND b ≥ 90    (second only)
D: a < 90 AND b < 90    (neither)

Alternative using simplified logic:
A: highA AND highB
B: highA AND NOT highB
C: NOT highA AND highB
D: NOT highA AND NOT highB
```

---

## Key Takeaways

1. **Logical AND (`&&`)** — both conditions must be true
2. **If-else chain** — mutually exclusive conditions
3. **Nested if** — can be more efficient (fewer comparisons)
4. **Threshold comparison** — `>=` includes boundary value
5. **Complete coverage** — all 4 cases must be handled

---

## Practice Exercises

1. Add grade **E** for when average is below 85
2. Handle **three** scores with 8 possible grades
3. Use **weighted average** instead of individual thresholds
4. Output **detailed reasoning** for the grade

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/102)*
