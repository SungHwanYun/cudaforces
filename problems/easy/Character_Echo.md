# Character Echo

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Given a lowercase letter a and a positive integer b, print the character a repeated b times, with each character separated by a single space.

### Input
A single line contains a lowercase letter a and an integer b, separated by a space.

**Constraints:**
- a is a lowercase English letter ('a' to 'z')
- 1 ≤ b ≤ 100

### Output
Print the character a repeated b times, separated by spaces.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| h 5 | h h h h h |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| a 1 | a |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| z 10 | z z z z z z z z z z |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void echoKernel(char* ch, int* count) {
    int idx = threadIdx.x;
    if (idx == 0) {
        char c = *ch;
        int n = *count;
        
        for (int i = 0; i < n; i++) {
            if (i > 0) printf(" ");
            printf("%c", c);
        }
        printf("\n");
    }
}

int main() {
    char ch;
    int count;
    scanf(" %c %d", &ch, &count);
    
    // Device memory
    char *d_ch;
    int *d_count;
    cudaMalloc(&d_ch, sizeof(char));
    cudaMalloc(&d_count, sizeof(int));
    
    // Copy inputs to device
    cudaMemcpy(d_ch, &ch, sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    echoKernel<<<1, 1>>>(d_ch, d_count);
    cudaDeviceSynchronize();
    
    cudaFree(d_ch);
    cudaFree(d_count);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void echoKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs repeated output on GPU |

---

## CUDA Concepts Covered

### 1. Loop-Based Output

Print the character n times with spaces between them:

```cuda
for (int i = 0; i < n; i++) {
    if (i > 0) printf(" ");  // Space before all except first
    printf("%c", c);
}
```

### 2. Space Handling

The key challenge is placing spaces correctly:

```
Output: "h h h h h"
         ↑ ↑ ↑ ↑
       spaces between characters, not at ends
```

**Two common patterns:**

```cuda
// Pattern 1: Space before (skip first)
for (int i = 0; i < n; i++) {
    if (i > 0) printf(" ");
    printf("%c", c);
}

// Pattern 2: Space after (skip last)
for (int i = 0; i < n; i++) {
    printf("%c", c);
    if (i < n - 1) printf(" ");
}
```

### 3. Visualization

```
Input: 'h', count = 5

Iteration:  0     1     2     3     4
            │     │     │     │     │
Output:    'h'  ' h'  ' h'  ' h'  ' h'
            │     │     │     │     │
            └──┬──┴──┬──┴──┬──┴──┬──┘
               └─────┴─────┴─────┘
                    
Result: "h h h h h"
```

### 4. Reading Character Input

```cuda
scanf(" %c %d", &ch, &count);
```

The space before `%c` is important — it skips any leading whitespace.

### 5. Edge Case: b = 1

```
Input: 'a', count = 1

Loop runs once:
  i = 0: i > 0? No → print 'a' only

Output: "a" (no spaces)
```

---

## Alternative Solutions

### Using Parallel Threads

```cuda
__global__ void echoKernel(char* ch, char* output, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        int pos = idx * 2;  // Each char takes 2 positions (char + space)
        output[pos] = *ch;
        if (idx < n - 1) {
            output[pos + 1] = ' ';
        }
    }
}

int main() {
    char ch;
    int count;
    scanf(" %c %d", &ch, &count);
    
    char h_output[201];  // Max: 100 chars + 99 spaces + null
    
    char *d_ch, *d_output;
    cudaMalloc(&d_ch, sizeof(char));
    cudaMalloc(&d_output, 201 * sizeof(char));
    
    cudaMemcpy(d_ch, &ch, sizeof(char), cudaMemcpyHostToDevice);
    
    echoKernel<<<1, count>>>(d_ch, d_output, count);
    
    int len = count * 2 - 1;  // chars + spaces
    cudaMemcpy(h_output, d_output, len * sizeof(char), cudaMemcpyDeviceToHost);
    h_output[len] = '\0';
    
    printf("%s\n", h_output);
    
    cudaFree(d_ch);
    cudaFree(d_output);
    
    return 0;
}
```

### Space After Pattern

```cuda
__global__ void echoKernel(char* ch, int* count) {
    int idx = threadIdx.x;
    if (idx == 0) {
        char c = *ch;
        int n = *count;
        
        for (int i = 0; i < n; i++) {
            printf("%c", c);
            if (i < n - 1) printf(" ");  // Space after except last
        }
        printf("\n");
    }
}
```

### First Character Special Case

```cuda
__global__ void echoKernel(char* ch, int* count) {
    int idx = threadIdx.x;
    if (idx == 0) {
        char c = *ch;
        int n = *count;
        
        printf("%c", c);  // First character
        for (int i = 1; i < n; i++) {
            printf(" %c", c);  // Space + character
        }
        printf("\n");
    }
}
```

---

## Output Format Patterns

| b | Output | Length |
|---|--------|--------|
| 1 | "a" | 1 |
| 2 | "a a" | 3 |
| 3 | "a a a" | 5 |
| 5 | "a a a a a" | 9 |
| n | ... | 2n - 1 |

**Formula:** Output length = 2×b - 1 (b characters + (b-1) spaces)

---

## Common Mistakes

### ❌ Extra Space at End
```cuda
for (int i = 0; i < n; i++) {
    printf("%c ", c);  // Wrong! Trailing space
}
```

### ❌ Extra Space at Start
```cuda
for (int i = 0; i < n; i++) {
    printf(" %c", c);  // Wrong! Leading space
}
```

### ❌ Missing scanf Space
```cuda
scanf("%c %d", &ch, &count);   // May read leftover whitespace
scanf(" %c %d", &ch, &count);  // Correct - space skips whitespace
```

### ❌ Wrong Format Specifier
```cuda
printf("%s", c);   // Wrong! %s is for strings
printf("%c", c);   // Correct - %c for single character
```

---

## Character vs String

| Type | Declaration | Format | Example |
|------|-------------|--------|---------|
| char | `char c = 'a';` | `%c` | 'a' |
| string | `char s[] = "abc";` | `%s` | "abc" |

This problem uses a single character, not a string.

---

## Key Takeaways

1. **Loop with conditional space** — handle separators correctly
2. **"Space before except first"** — common pattern for separators
3. **scanf whitespace** — use `" %c"` to skip leading whitespace
4. **Edge case b=1** — no spaces needed
5. **Output length** — 2n-1 for n items with spaces

---

## Practice Exercises

1. Print in **reverse order** (same characters, but right to left)
2. Print with **comma separators** instead of spaces
3. Print each character on a **new line**
4. Alternate between **two characters** (e.g., "a b a b a")

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/124)*
