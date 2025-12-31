# String Echo

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Given a string a and a positive integer b, print the string a repeated b times, with each occurrence separated by a single space.

### Input
The first line contains an integer n (the length of string a) and the string a, separated by a space.

The second line contains an integer b.

The string a consists only of uppercase and lowercase English letters.

**Constraints:**
- 1 ≤ n ≤ 10
- 1 ≤ b ≤ 100

### Output
Print the string a repeated b times, separated by spaces.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 10 cudaforces<br>3 | cudaforces cudaforces cudaforces |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 2 Hi<br>5 | Hi Hi Hi Hi Hi |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 4 CUDA<br>1 | CUDA |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void stringEchoKernel(char* str, int* count) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int n = *count;
        
        for (int i = 0; i < n; i++) {
            if (i > 0) printf(" ");
            printf("%s", str);
        }
        printf("\n");
    }
}

int main() {
    int n, b;
    char h_str[11];  // Max length 10 + null terminator
    
    scanf("%d %s", &n, h_str);
    scanf("%d", &b);
    
    // Device memory
    char *d_str;
    int *d_count;
    cudaMalloc(&d_str, (n + 1) * sizeof(char));
    cudaMalloc(&d_count, sizeof(int));
    
    // Copy inputs to device
    cudaMemcpy(d_str, h_str, (n + 1) * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &b, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    stringEchoKernel<<<1, 1>>>(d_str, d_count);
    cudaDeviceSynchronize();
    
    cudaFree(d_str);
    cudaFree(d_count);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void stringEchoKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs repeated string output on GPU |

---

## CUDA Concepts Covered

### 1. String vs Character Echo

The pattern is identical to Character Echo, but uses `%s` instead of `%c`:

```cuda
// Character Echo
printf("%c", c);

// String Echo
printf("%s", str);
```

### 2. Loop-Based Output

Print the string n times with spaces between them:

```cuda
for (int i = 0; i < n; i++) {
    if (i > 0) printf(" ");  // Space before all except first
    printf("%s", str);
}
```

### 3. Visualization

```
Input: "Hi", count = 5

Iteration:  0      1      2      3      4
            │      │      │      │      │
Output:   "Hi"  " Hi"  " Hi"  " Hi"  " Hi"
            │      │      │      │      │
            └──────┴──────┴──────┴──────┘
                    
Result: "Hi Hi Hi Hi Hi"
```

### 4. Memory Allocation for Strings

```cuda
// Allocate space for string + null terminator
cudaMalloc(&d_str, (n + 1) * sizeof(char));

// Copy including null terminator
cudaMemcpy(d_str, h_str, (n + 1) * sizeof(char), cudaMemcpyHostToDevice);
```

### 5. Reading String Input

```cuda
scanf("%d %s", &n, h_str);  // Read length and string
scanf("%d", &b);            // Read repeat count
```

---

## Alternative Solutions

### First String Special Case

```cuda
__global__ void stringEchoKernel(char* str, int* count) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int n = *count;
        
        printf("%s", str);  // First string
        for (int i = 1; i < n; i++) {
            printf(" %s", str);  // Space + string
        }
        printf("\n");
    }
}
```

### Space After Pattern

```cuda
__global__ void stringEchoKernel(char* str, int* count) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int n = *count;
        
        for (int i = 0; i < n; i++) {
            printf("%s", str);
            if (i < n - 1) printf(" ");  // Space after except last
        }
        printf("\n");
    }
}
```

### Building Output String

```cuda
__global__ void stringEchoKernel(char* str, int strLen, int* count, char* output) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int n = *count;
        int pos = 0;
        
        for (int i = 0; i < n; i++) {
            if (i > 0) {
                output[pos++] = ' ';
            }
            for (int j = 0; j < strLen; j++) {
                output[pos++] = str[j];
            }
        }
        output[pos] = '\0';
    }
}
```

---

## Comparison: Character vs String Echo

| Aspect | Character Echo | String Echo |
|--------|---------------|-------------|
| Input type | Single char | String |
| Format specifier | `%c` | `%s` |
| Memory allocation | 1 byte | n+1 bytes |
| Output per iteration | 1 char | n chars |

---

## Output Format

| b | String "Hi" | Output Length |
|---|-------------|---------------|
| 1 | "Hi" | 2 |
| 2 | "Hi Hi" | 5 |
| 3 | "Hi Hi Hi" | 8 |
| 5 | "Hi Hi Hi Hi Hi" | 14 |

**Formula:** Output length = b × len(str) + (b-1) spaces

For string length L and count b: **b × L + (b - 1)**

---

## Common Mistakes

### ❌ Extra Space at End
```cuda
for (int i = 0; i < n; i++) {
    printf("%s ", str);  // Wrong! Trailing space
}
```

### ❌ Forgetting Null Terminator
```cuda
cudaMalloc(&d_str, n * sizeof(char));      // Wrong! Missing space for '\0'
cudaMalloc(&d_str, (n + 1) * sizeof(char)); // Correct
```

### ❌ Using %c for String
```cuda
printf("%c", str);   // Wrong! %c is for single character
printf("%s", str);   // Correct - %s for strings
```

### ❌ Wrong Input Parsing
```cuda
scanf("%s %d", h_str, &n);  // Wrong order!
scanf("%d %s", &n, h_str);  // Correct - length first, then string
```

---

## String Memory Layout

```
String: "CUDA" (length 4)

Index:   0    1    2    3    4
       ┌────┬────┬────┬────┬────┐
       │ 'C'│ 'U'│ 'D'│ 'A'│'\0'│
       └────┴────┴────┴────┴────┘
                           ↑
                    Null terminator
                    
Allocation: 5 bytes (4 chars + 1 null)
```

---

## Key Takeaways

1. **Same pattern as Character Echo** — only format specifier changes
2. **String memory** — always include space for null terminator
3. **%s format** — automatically handles string until null terminator
4. **Space handling** — "space before except first" pattern
5. **Output length** — b × L + (b - 1) for string length L

---

## Practice Exercises

1. Print each string on a **new line** instead of spaces
2. Print strings in **reverse order** (last to first)
3. Alternate between **two different strings**
4. Add **numbering** before each string (1. CUDA 2. CUDA 3. CUDA)

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/125)*
