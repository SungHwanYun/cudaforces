# Echo String

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given a string s consisting of lowercase English letters. Print the string s.

### Input
The first line contains a string s consisting of lowercase English letters.

It is guaranteed that 1 ≤ length of s ≤ 100.

### Output
Print the string s on the first line.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| hello | hello |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| world | world |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| cuda | cuda |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void echoKernel(char* str, int len) {
    int idx = threadIdx.x;
    if (idx == 0) {
        for (int i = 0; i < len; i++) {
            printf("%c", str[i]);
        }
        printf("\n");
    }
}

int main() {
    char h_str[101];
    scanf("%s", h_str);
    
    // Calculate string length manually (no strlen allowed)
    int len = 0;
    while (h_str[len] != '\0') {
        len++;
    }
    
    // Device memory
    char* d_str;
    cudaMalloc(&d_str, (len + 1) * sizeof(char));
    
    // Copy to device
    cudaMemcpy(d_str, h_str, (len + 1) * sizeof(char), cudaMemcpyHostToDevice);
    
    // Launch kernel
    echoKernel<<<1, 1>>>(d_str, len);
    
    cudaDeviceSynchronize();
    cudaFree(d_str);
    
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
| ✅ Meaningful computation | Iterates through string and prints |

---

## CUDA Concepts Covered

### 1. String Handling in CUDA

Strings are character arrays terminated by `'\0'`:

```cuda
char h_str[101];  // Fixed-size buffer (max 100 chars + null terminator)
scanf("%s", h_str);
```

> ⚠️ **Note**: `std::string` is prohibited. Use C-style char arrays.

### 2. Manual String Length Calculation

Since `strlen()` is forbidden, calculate length manually:

```cuda
int len = 0;
while (h_str[len] != '\0') {
    len++;
}
```

This iterates until the null terminator is found.

### 3. Memory Layout for Strings

```
String "hello":

Index:    [0]  [1]  [2]  [3]  [4]  [5]
Content:  'h'  'e'  'l'  'l'  'o'  '\0'
                                    ↑
                              Null terminator

Allocate len + 1 bytes to include the null terminator.
```

### 4. Copying Strings to Device

```cuda
// Allocate space for string + null terminator
cudaMalloc(&d_str, (len + 1) * sizeof(char));

// Copy entire string including null terminator
cudaMemcpy(d_str, h_str, (len + 1) * sizeof(char), cudaMemcpyHostToDevice);
```

### 5. Printing Strings in Kernel

Two approaches to print from device:

```cuda
// Approach 1: Loop through characters
for (int i = 0; i < len; i++) {
    printf("%c", str[i]);
}
printf("\n");

// Approach 2: Print entire string (if null-terminated)
printf("%s\n", str);
```

---

## Alternative Solutions

### Using %s Format Specifier

```cuda
__global__ void echoKernel(char* str) {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("%s\n", str);  // Print entire null-terminated string
    }
}

int main() {
    char h_str[101];
    scanf("%s", h_str);
    
    int len = 0;
    while (h_str[len] != '\0') len++;
    
    char* d_str;
    cudaMalloc(&d_str, (len + 1) * sizeof(char));
    cudaMemcpy(d_str, h_str, (len + 1) * sizeof(char), cudaMemcpyHostToDevice);
    
    echoKernel<<<1, 1>>>(d_str);
    
    cudaDeviceSynchronize();
    cudaFree(d_str);
    
    return 0;
}
```

### Parallel Character Output (Educational)

```cuda
__global__ void echoParallelKernel(char* str, int len) {
    int idx = threadIdx.x;
    
    // Each thread handles one character
    // Note: Order not guaranteed in parallel!
    if (idx < len) {
        // This is for educational purposes only
        // Actual output order may vary
    }
}
```

> ⚠️ **Warning**: Parallel printing doesn't guarantee character order. For correct string output, use single-threaded printing.

### Using fgets for Strings with Spaces

```cuda
// If input might contain spaces (not in this problem):
char h_str[101];
fgets(h_str, 101, stdin);

// Remove trailing newline if present
int len = 0;
while (h_str[len] != '\0') len++;
if (len > 0 && h_str[len - 1] == '\n') {
    h_str[len - 1] = '\0';
    len--;
}
```

---

## Common Mistakes

### ❌ Using strlen()
```cuda
int len = strlen(h_str);  // FORBIDDEN! strlen is not allowed
```
Calculate length manually with a loop.

### ❌ Using std::string
```cuda
std::string str;  // FORBIDDEN! Use char arrays instead
std::cin >> str;
```

### ❌ Forgetting Null Terminator Space
```cuda
cudaMalloc(&d_str, len * sizeof(char));  // Wrong! Missing space for '\0'
cudaMalloc(&d_str, (len + 1) * sizeof(char));  // Correct!
```

### ❌ Buffer Overflow
```cuda
char h_str[10];  // Too small for 100 character input!
scanf("%s", h_str);  // Potential buffer overflow
```
Always allocate enough space (101 for max 100 chars).

### ❌ Missing cudaDeviceSynchronize
```cuda
echoKernel<<<1, 1>>>(d_str, len);
// Missing cudaDeviceSynchronize()!
return 0;  // Program may exit before printf completes
```

---

## String Operations Reference

Since many string functions are forbidden, here are manual implementations:

### String Length
```cuda
int stringLength(char* str) {
    int len = 0;
    while (str[len] != '\0') len++;
    return len;
}
```

### String Copy
```cuda
void stringCopy(char* dest, char* src) {
    int i = 0;
    while (src[i] != '\0') {
        dest[i] = src[i];
        i++;
    }
    dest[i] = '\0';
}
```

### String Compare
```cuda
int stringCompare(char* s1, char* s2) {
    int i = 0;
    while (s1[i] != '\0' && s2[i] != '\0') {
        if (s1[i] != s2[i]) return s1[i] - s2[i];
        i++;
    }
    return s1[i] - s2[i];
}
```

---

## Character Array vs Integer Array

| Aspect | char Array | int Array |
|--------|------------|-----------|
| Element size | 1 byte | 4 bytes |
| Terminator | `'\0'` (null) | None (need length) |
| Print format | `%c` or `%s` | `%d` |
| Memory | `len + 1` bytes | `len * 4` bytes |

---

## Key Takeaways

1. **Strings are char arrays** terminated by `'\0'`
2. **Manual length calculation** — `strlen()` is forbidden
3. **Allocate len + 1** for null terminator
4. **`%s` format** works in CUDA printf for null-terminated strings
5. **Sequential printing** maintains character order

---

## Practice Exercises

1. Read a string and print it in **reverse**
2. Read a string and print each character on a **new line**
3. Count the number of vowels in the string
4. Convert lowercase to **uppercase** and print

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/18)*
