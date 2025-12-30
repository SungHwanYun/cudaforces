# Student Grade Calculator

| Difficulty | Memory Limit | Time Limit |
|------------|--------------|------------|
| Easy | 128 MB | 1 s |

## Problem Description

Calculate a student's total score and average score across five subjects.

### Input

A single line containing six space-separated integers: **k m e s c t**

- k: Korean score
- m: Math score
- e: English score
- s: Science score
- c: Coding score
- t: Student ID number

**Constraints:**
- 1 ≤ k, m, e, s, c ≤ 100
- k + m + e + s + c is divisible by 5
- 1 ≤ t ≤ 1000

### Output

Print two space-separated integers: the total score and the average score.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 90 85 91 90 89 1 | 445 89 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 90 85 80 95 100 12 | 450 90 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 70 60 80 75 65 7 | 350 70 |

---

## Solution Code

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](https://github.com/SungHwanYun/cudaforces/blob/main/GUIDE.md) first.

```cuda
__global__ void calculateGrade(int* scores, int* result) {
    int idx = threadIdx.x;
    
    // Thread 0: Calculate total
    if (idx == 0) {
        int total = 0;
        for (int i = 0; i < 5; i++) {
            total += scores[i];
        }
        result[0] = total;
        result[1] = total / 5;
    }
}

int main() {
    int k, m, e, s, c, t;
    scanf("%d %d %d %d %d %d", &k, &m, &e, &s, &c, &t);
    
    int h_scores[5] = {k, m, e, s, c};
    int h_result[2];
    
    int *d_scores, *d_result;
    cudaMalloc(&d_scores, 5 * sizeof(int));
    cudaMalloc(&d_result, 2 * sizeof(int));
    
    cudaMemcpy(d_scores, h_scores, 5 * sizeof(int), cudaMemcpyHostToDevice);
    
    calculateGrade<<<1, 1>>>(d_scores, d_result);
    
    cudaMemcpy(h_result, d_result, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d %d\n", h_result[0], h_result[1]);
    
    cudaFree(d_scores);
    cudaFree(d_result);
    
    return 0;
}
```

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void calculateGrade()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launch syntax |
| ✅ Meaningful computation | Sum and average calculation |

---

## CUDA Concepts Covered

### 1. Array Data Transfer

Transfer multiple values between host and device using arrays:

```cuda
int h_scores[5] = {k, m, e, s, c};  // Host array
int *d_scores;
cudaMalloc(&d_scores, 5 * sizeof(int));
cudaMemcpy(d_scores, h_scores, 5 * sizeof(int), cudaMemcpyHostToDevice);
```

### 2. Kernel with Array Parameters

Kernels can receive pointer parameters to access device memory:

```cuda
__global__ void calculateGrade(int* scores, int* result) {
    // Access scores[0], scores[1], etc.
    // Write to result[0], result[1]
}
```

### 3. Single-Thread Computation

For simple sequential calculations, a single thread is sufficient:

```cuda
calculateGrade<<<1, 1>>>(d_scores, d_result);
```

While this doesn't utilize GPU parallelism, it satisfies CUDA OJ validation requirements.

---

## Common Mistakes

### ❌ Forgetting to Copy Input Data
```cuda
// Missing cudaMemcpy before kernel
calculateGrade<<<1, 1>>>(d_scores, d_result);  // d_scores contains garbage!
```

### ❌ Wrong Array Size in cudaMemcpy
```cuda
cudaMemcpy(d_scores, h_scores, sizeof(int), cudaMemcpyHostToDevice);
// Only copies 1 int instead of 5!
```

### ❌ Integer Division Issues
```cuda
result[1] = total / 5;  // OK for this problem (guaranteed divisible)
// For general case, consider: result[1] = (total + 2) / 5; for rounding
```

---

## Key Takeaways

1. **Arrays can be transferred** to GPU using `cudaMemcpy` with proper size
2. **Multiple results** can be returned via output array
3. **Simple computations** may use single thread but must follow CUDA patterns
4. **Size calculation**: `n * sizeof(type)` for n elements

---

## Practice Exercises

1. Modify to use 5 threads, each adding one score to a shared sum
2. Add validation to check if scores are within valid range
3. Calculate and output the maximum and minimum scores as well

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/5)*
