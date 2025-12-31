# Character Cipher

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

A character transformation table T is given as follows:

```
T = {'a': 'T', 'b': 'C', 'c': 'P', 'd': 'Q', 'e': 'A'}
```

That is, character 'a' transforms to 'T', 'b' transforms to 'C', 'c' transforms to 'P', 'd' transforms to 'Q', and 'e' transforms to 'A'.

Given a string S, transform each character using the table T and print the resulting string.

### Input
The first line contains an integer n, the length of string S.

The second line contains the string S.

The string S consists only of characters 'a', 'b', 'c', 'd', 'e'.

**Constraints:**
- 1 ≤ n ≤ 100

### Output
Print the transformed string.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 5<br>abcde | TCPQA |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 3<br>aaa | TTT |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 10<br>abcdeedcba | TCPQAAQPCT |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void cipherKernel(char* input, char* output, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        char c = input[idx];
        char result;
        
        if (c == 'a') result = 'T';
        else if (c == 'b') result = 'C';
        else if (c == 'c') result = 'P';
        else if (c == 'd') result = 'Q';
        else result = 'A';  // c == 'e'
        
        output[idx] = result;
    }
}

int main() {
    int n;
    char h_input[101], h_output[101];
    
    scanf("%d", &n);
    scanf("%s", h_input);
    
    // Device memory
    char *d_input, *d_output;
    cudaMalloc(&d_input, (n + 1) * sizeof(char));
    cudaMalloc(&d_output, (n + 1) * sizeof(char));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, (n + 1) * sizeof(char), cudaMemcpyHostToDevice);
    
    // Launch kernel - one thread per character
    cipherKernel<<<1, n>>>(d_input, d_output, n);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, (n + 1) * sizeof(char), cudaMemcpyDeviceToHost);
    h_output[n] = '\0';
    
    printf("%s\n", h_output);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void cipherKernel()` |
| ✅ Uses parallelism | Each thread transforms one character |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, n>>>` launches n threads |
| ✅ Meaningful computation | Performs character substitution on GPU |

---

## CUDA Concepts Covered

### 1. Substitution Cipher

A substitution cipher replaces each character with another according to a fixed mapping:

```
Input:  a  b  c  d  e
        ↓  ↓  ↓  ↓  ↓
Output: T  C  P  Q  A
```

### 2. Parallel Character Processing

Each thread handles one character independently:

```cuda
cipherKernel<<<1, n>>>(d_input, d_output, n);
```

```
Thread 0: input[0] → output[0]
Thread 1: input[1] → output[1]
Thread 2: input[2] → output[2]
...
```

### 3. Visualization

```
Input String: "abcde"

Thread:   0     1     2     3     4
          │     │     │     │     │
Input:   'a'   'b'   'c'   'd'   'e'
          │     │     │     │     │
          ↓     ↓     ↓     ↓     ↓
Output:  'T'   'C'   'P'   'Q'   'A'

Result: "TCPQA"
```

### 4. Lookup Table Implementation

```cuda
// Method 1: If-else chain
if (c == 'a') result = 'T';
else if (c == 'b') result = 'C';
else if (c == 'c') result = 'P';
else if (c == 'd') result = 'Q';
else result = 'A';

// Method 2: Array lookup
char table[] = "TCPQA";  // Index: a=0, b=1, c=2, d=3, e=4
result = table[c - 'a'];
```

### 5. Thread Indexing

```cuda
int idx = threadIdx.x;
if (idx < n) {
    // Process character at position idx
}
```

The bounds check `idx < n` ensures threads don't access invalid memory.

---

## Alternative Solutions

### Using Array Lookup

```cuda
__global__ void cipherKernel(char* input, char* output, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        // Lookup table: index 0='a'→'T', 1='b'→'C', etc.
        char table[] = "TCPQA";
        output[idx] = table[input[idx] - 'a'];
    }
}
```

### Using Switch Statement

```cuda
__global__ void cipherKernel(char* input, char* output, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        char c = input[idx];
        switch (c) {
            case 'a': output[idx] = 'T'; break;
            case 'b': output[idx] = 'C'; break;
            case 'c': output[idx] = 'P'; break;
            case 'd': output[idx] = 'Q'; break;
            case 'e': output[idx] = 'A'; break;
        }
    }
}
```

### Sequential Processing (Single Thread)

```cuda
__global__ void cipherKernel(char* input, char* output, int n) {
    int idx = threadIdx.x;
    if (idx == 0) {
        char table[] = "TCPQA";
        for (int i = 0; i < n; i++) {
            output[i] = table[input[i] - 'a'];
        }
    }
}
```

---

## Transformation Table

| Input | Output | Index |
|-------|--------|-------|
| 'a' | 'T' | 0 |
| 'b' | 'C' | 1 |
| 'c' | 'P' | 2 |
| 'd' | 'Q' | 3 |
| 'e' | 'A' | 4 |

### Character Arithmetic

```cuda
// Convert 'a'-'e' to index 0-4
int index = c - 'a';

// Examples:
// 'a' - 'a' = 0
// 'b' - 'a' = 1
// 'e' - 'a' = 4
```

---

## Common Mistakes

### ❌ Missing Bounds Check
```cuda
// Wrong! May access invalid memory
output[idx] = table[input[idx] - 'a'];

// Correct
if (idx < n) {
    output[idx] = table[input[idx] - 'a'];
}
```

### ❌ Forgetting Null Terminator
```cuda
// After copying back to host
h_output[n] = '\0';  // Important for printf("%s")
```

### ❌ Wrong Table Order
```cuda
char table[] = "ACPQT";  // Wrong! Order must match a,b,c,d,e
char table[] = "TCPQA";  // Correct
```

### ❌ Incorrect Character Arithmetic
```cuda
output[idx] = table[input[idx]];       // Wrong! Index would be ~97
output[idx] = table[input[idx] - 'a']; // Correct - normalize to 0-4
```

---

## Parallel vs Sequential

| Approach | Threads | Time Complexity |
|----------|---------|-----------------|
| Parallel | n threads | O(1) per thread |
| Sequential | 1 thread | O(n) in loop |

For this problem (n ≤ 100), both are fast enough, but parallel demonstrates CUDA's power.

---

## Palindrome Property (Example 3)

```
Input:  abcdeedcba (palindrome)
Output: TCPQAAQPCT (also palindrome!)

Why? Each character maps to exactly one output character.
If input is symmetric, output is also symmetric.
```

---

## Key Takeaways

1. **Substitution cipher** — fixed mapping from input to output characters
2. **Parallel processing** — one thread per character
3. **Array lookup** — efficient O(1) transformation
4. **Character arithmetic** — `c - 'a'` converts to 0-based index
5. **Bounds checking** — essential for safe parallel access

---

## Practice Exercises

1. Implement the **reverse cipher** (decode: T→a, C→b, etc.)
2. Create a **Caesar cipher** with configurable shift
3. Handle **uppercase and lowercase** characters
4. Process strings **longer than 1024** characters using multiple blocks

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/121)*
