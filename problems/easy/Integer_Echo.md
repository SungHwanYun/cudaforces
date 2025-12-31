# Integer Echo

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Given an integer a and a positive integer b, print the integer a repeated b times, with each occurrence separated by a single space.

### Input
The first line contains an integer a.

The second line contains an integer b.

**Constraints:**
- 1 ≤ a ≤ 100
- 1 ≤ b ≤ 100

### Output
Print the integer a repeated b times, separated by spaces.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 42<br>3 | 42 42 42 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 7<br>5 | 7 7 7 7 7 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 100<br>1 | 100 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void integerEchoKernel(int* num, int* count) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int a = *num;
        int b = *count;
        
        for (int i = 0; i < b; i++) {
            if (i > 0) printf(" ");
            printf("%d", a);
        }
        printf("\n");
    }
}

int main() {
    int a, b;
    scanf("%d", &a);
    scanf("%d", &b);
    
    // Device memory
    int *d_num, *d_count;
    cudaMalloc(&d_num, sizeof(int));
    cudaMalloc(&d_count, sizeof(int));
    
    // Copy inputs to device
    cudaMemcpy(d_num, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &b, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    integerEchoKernel<<<1, 1>>>(d_num, d_count);
    cudaDeviceSynchronize();
    
    cudaFree(d_num);
    cudaFree(d_count);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void integerEchoKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs repeated integer output on GPU |

---

## CUDA Concepts Covered

### 1. Integer Echo Pattern

Same pattern as Character and String Echo, using `%d` format:

```cuda
for (int i = 0; i < b; i++) {
    if (i > 0) printf(" ");  // Space before all except first
    printf("%d", a);
}
```

### 2. Visualization

```
Input: a = 42, b = 3

Iteration:  0       1       2
            │       │       │
Output:   "42"   " 42"   " 42"
            │       │       │
            └───────┴───────┘
                    
Result: "42 42 42"
```

### 3. Format Specifier Comparison

| Type | Format | Example |
|------|--------|---------|
| Character | `%c` | 'h' |
| String | `%s` | "Hi" |
| Integer | `%d` | 42 |

### 4. Integer Memory

Unlike strings, integers don't need null terminators:

```cuda
// Integer: just sizeof(int) bytes
cudaMalloc(&d_num, sizeof(int));

// No null terminator needed
cudaMemcpy(d_num, &a, sizeof(int), cudaMemcpyHostToDevice);
```

### 5. Variable-Width Output

Integers have variable character width:

```
a = 7   → "7"     (1 character)
a = 42  → "42"    (2 characters)
a = 100 → "100"   (3 characters)
```

---

## Alternative Solutions

### First Number Special Case

```cuda
__global__ void integerEchoKernel(int* num, int* count) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int a = *num;
        int b = *count;
        
        printf("%d", a);  // First number
        for (int i = 1; i < b; i++) {
            printf(" %d", a);  // Space + number
        }
        printf("\n");
    }
}
```

### Space After Pattern

```cuda
__global__ void integerEchoKernel(int* num, int* count) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int a = *num;
        int b = *count;
        
        for (int i = 0; i < b; i++) {
            printf("%d", a);
            if (i < b - 1) printf(" ");  // Space after except last
        }
        printf("\n");
    }
}
```

### Using Single cudaMalloc

```cuda
__global__ void integerEchoKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int a = data[0];
        int b = data[1];
        
        for (int i = 0; i < b; i++) {
            if (i > 0) printf(" ");
            printf("%d", a);
        }
        printf("\n");
    }
}

int main() {
    int a, b;
    scanf("%d", &a);
    scanf("%d", &b);
    
    int h_data[2] = {a, b};
    
    int* d_data;
    cudaMalloc(&d_data, 2 * sizeof(int));
    cudaMemcpy(d_data, h_data, 2 * sizeof(int), cudaMemcpyHostToDevice);
    
    integerEchoKernel<<<1, 1>>>(d_data);
    cudaDeviceSynchronize();
    
    cudaFree(d_data);
    
    return 0;
}
```

---

## Echo Series Comparison

| Problem | Input Type | Format | Memory |
|---------|-----------|--------|--------|
| Character Echo | char | `%c` | 1 byte |
| String Echo | char[] | `%s` | n+1 bytes |
| **Integer Echo** | int | `%d` | 4 bytes |

All three use the same output pattern:
```cuda
for (int i = 0; i < count; i++) {
    if (i > 0) printf(" ");
    printf(FORMAT, value);
}
```

---

## Common Mistakes

### ❌ Extra Space at End
```cuda
for (int i = 0; i < b; i++) {
    printf("%d ", a);  // Wrong! Trailing space
}
```

### ❌ Using Wrong Format
```cuda
printf("%c", a);   // Wrong! %c is for characters
printf("%s", a);   // Wrong! %s is for strings
printf("%d", a);   // Correct - %d for integers
```

### ❌ Mixing Up Variables
```cuda
printf("%d", b);   // Wrong! Should print 'a', not 'b'
printf("%d", a);   // Correct
```

---

## Output Examples

| a | b | Output | Character Count |
|---|---|--------|-----------------|
| 7 | 5 | "7 7 7 7 7" | 9 |
| 42 | 3 | "42 42 42" | 8 |
| 100 | 1 | "100" | 3 |
| 100 | 4 | "100 100 100 100" | 15 |

### Character Count Formula

For integer with d digits repeated b times:
- Total = b × d + (b - 1) spaces
- = b × d + b - 1
- = b × (d + 1) - 1

---

## Key Takeaways

1. **Same pattern** — echoing works identically for char, string, and int
2. **Format specifier** — use `%d` for integers
3. **No null terminator** — integers are simpler than strings
4. **Variable width** — output length depends on number of digits
5. **Unified approach** — "space before except first" pattern works for all types

---

## Practice Exercises

1. Echo **floating-point** numbers with `%f` or `%.2f`
2. Echo with **comma separators** instead of spaces
3. Print numbers in **descending order** (42 41 40 ... for count times, decrementing)
4. Alternate between **two different integers**

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/126)*
