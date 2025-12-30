# Letter Grade

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Given a programming score p, determine the corresponding letter grade.

The grading criteria are as follows:

- 90 ≤ p ≤ 100 : **A**
- 80 ≤ p ≤ 89 : **B**
- 70 ≤ p ≤ 79 : **C**
- 60 ≤ p ≤ 69 : **D**
- p ≤ 59 : **E**

### Input
A single integer p representing the score.

**Constraints:**
- 1 ≤ p ≤ 100

### Output
Print the corresponding letter grade.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 90 | A |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 75 | C |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 42 | E |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void letterGradeKernel(int* score, char* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int p = *score;
        
        if (p >= 90) {
            *result = 'A';
        } else if (p >= 80) {
            *result = 'B';
        } else if (p >= 70) {
            *result = 'C';
        } else if (p >= 60) {
            *result = 'D';
        } else {
            *result = 'E';
        }
    }
}

int main() {
    int p;
    scanf("%d", &p);
    
    // Device memory
    int *d_score;
    char *d_result;
    cudaMalloc(&d_score, sizeof(int));
    cudaMalloc(&d_result, sizeof(char));
    
    // Copy input to device
    cudaMemcpy(d_score, &p, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    letterGradeKernel<<<1, 1>>>(d_score, d_result);
    
    // Copy result back to host
    char result;
    cudaMemcpy(&result, d_result, sizeof(char), cudaMemcpyDeviceToHost);
    
    printf("%c\n", result);
    
    cudaFree(d_score);
    cudaFree(d_result);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void letterGradeKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs range-based classification on GPU |

---

## CUDA Concepts Covered

### 1. Cascading If-Else

When checking ranges, order from highest to lowest:

```cuda
if (p >= 90) {
    *result = 'A';
} else if (p >= 80) {   // Already know p < 90
    *result = 'B';
} else if (p >= 70) {   // Already know p < 80
    *result = 'C';
} else if (p >= 60) {   // Already know p < 70
    *result = 'D';
} else {                // p < 60
    *result = 'E';
}
```

### 2. Implicit Upper Bounds

The `else if` structure creates implicit upper bounds:

```
if (p >= 90)       → 90-100 (upper bound is constraint)
else if (p >= 80)  → 80-89  (upper bound is implicit from failed p >= 90)
else if (p >= 70)  → 70-79  (upper bound is implicit from failed p >= 80)
else if (p >= 60)  → 60-69  (upper bound is implicit from failed p >= 70)
else               → 1-59   (lower bound is constraint)
```

### 3. Range Visualization

```
Score:  0   10   20   30   40   50   60   70   80   90   100
        |____|____|____|____|____|____|____|____|____|____|
        <─────────── E ──────────>│<─D─>│<─C─>│<─B─>│<─A─>│
                              59  60  69 70  79 80  89 90 100
```

### 4. Decision Flow

```
     ┌──────────────────┐
     │   p >= 90 ?      │
     └────────┬─────────┘
          YES │ NO
              ├──────────────────────────┐
              ▼                          ▼
            'A'              ┌──────────────────┐
                             │   p >= 80 ?      │
                             └────────┬─────────┘
                                  YES │ NO
                                      ├──────────────────────────┐
                                      ▼                          ▼
                                    'B'              ┌──────────────────┐
                                                     │   p >= 70 ?      │
                                                     └────────┬─────────┘
                                                          YES │ NO
                                                              ├──────────────┐
                                                              ▼              ▼
                                                            'C'    ┌──────────────┐
                                                                   │  p >= 60 ?   │
                                                                   └──────┬───────┘
                                                                      YES │ NO
                                                                          ├────────┐
                                                                          ▼        ▼
                                                                        'D'      'E'
```

### 5. Example Walkthrough

**Example 1**: `p = 90`
```
Check: 90 >= 90 → true → 'A'
```

**Example 2**: `p = 75`
```
Check: 75 >= 90 → false → next
Check: 75 >= 80 → false → next
Check: 75 >= 70 → true → 'C'
```

**Example 3**: `p = 42`
```
Check: 42 >= 90 → false → next
Check: 42 >= 80 → false → next
Check: 42 >= 70 → false → next
Check: 42 >= 60 → false → next
Else: 'E'
```

---

## Alternative Solutions

### Using Division (Tens Digit)

```cuda
__global__ void letterGradeKernel(int* score, char* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int p = *score;
        int tens = p / 10;
        
        if (tens >= 9) {
            *result = 'A';
        } else if (tens == 8) {
            *result = 'B';
        } else if (tens == 7) {
            *result = 'C';
        } else if (tens == 6) {
            *result = 'D';
        } else {
            *result = 'E';
        }
    }
}
```

### Using Array Lookup

```cuda
__global__ void letterGradeKernel(int* score, char* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // Grade lookup table indexed by tens digit
        // Index: 0-5=E, 6=D, 7=C, 8=B, 9-10=A
        char grades[] = "EEEEEEDCBAA";  // Index 0-10
        int tens = *score / 10;
        if (tens > 10) tens = 10;
        *result = grades[tens];
    }
}
```

### Using Arithmetic

```cuda
__global__ void letterGradeKernel(int* score, char* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int p = *score;
        // Map 90-100→0, 80-89→1, 70-79→2, 60-69→3, <60→4
        int grade;
        if (p >= 90) grade = 0;
        else if (p >= 60) grade = (89 - p) / 10 + 1;
        else grade = 4;
        
        *result = 'A' + grade;  // 'A'+0='A', 'A'+1='B', etc.
    }
}
```

---

## Grade Range Reference

| Grade | Min Score | Max Score | Range Size |
|-------|-----------|-----------|------------|
| A | 90 | 100 | 11 points |
| B | 80 | 89 | 10 points |
| C | 70 | 79 | 10 points |
| D | 60 | 69 | 10 points |
| E | 1 | 59 | 59 points |

---

## Common Mistakes

### ❌ Wrong Order (Low to High)
```cuda
// Wrong order - everyone gets E!
if (p >= 60) *result = 'D';      // 90 >= 60 is true!
else if (p >= 70) *result = 'C'; // Never reached for 90
else if (p >= 80) *result = 'B'; // Never reached for 90
else if (p >= 90) *result = 'A'; // Never reached for 90
```

### ❌ Missing else if
```cuda
if (p >= 90) *result = 'A';
if (p >= 80) *result = 'B';  // Overwrites A for p=95!
if (p >= 70) *result = 'C';  // Overwrites B!
```

### ❌ Redundant Range Check
```cuda
// Unnecessarily verbose
if (p >= 90 && p <= 100) *result = 'A';
else if (p >= 80 && p <= 89) *result = 'B';
// The upper bound is implicit from the else!
```

### ❌ Off-by-One Errors
```cuda
if (p > 90) *result = 'A';   // Wrong! 90 gets B
if (p >= 90) *result = 'A';  // Correct
```

---

## Boundary Test Cases

| Score | Grade | Reason |
|-------|-------|--------|
| 100 | A | Maximum score |
| 90 | A | Lower boundary of A |
| 89 | B | Upper boundary of B |
| 80 | B | Lower boundary of B |
| 79 | C | Upper boundary of C |
| 70 | C | Lower boundary of C |
| 69 | D | Upper boundary of D |
| 60 | D | Lower boundary of D |
| 59 | E | Upper boundary of E |
| 1 | E | Minimum score |

---

## Key Takeaways

1. **Order matters** — check from highest to lowest threshold
2. **Implicit bounds** — `else if` creates upper bounds automatically
3. **Boundary values** — use `>=` to include exact thresholds
4. **Cascading logic** — each condition eliminates possibilities
5. **Efficient structure** — at most 4 comparisons needed

---

## Practice Exercises

1. Add **plus/minus** grades (A+, A, A-, B+, etc.)
2. Convert **letter grade to GPA** (A=4.0, B=3.0, etc.)
3. Handle **invalid scores** (negative or > 100)
4. Calculate grade from **multiple test scores** with weights

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/103)*
