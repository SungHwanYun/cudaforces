# Print a Character

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given a single English letter. Print the character exactly as it appears in the input.

### Input
A single uppercase or lowercase English letter.

### Output
Output the given character without any modification.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| h | h |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| A | A |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| z | z |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void printCharKernel(char* c) {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("%c\n", *c);
    }
}

int main() {
    char ch;
    scanf(" %c", &ch);
    
    // Device memory
    char* d_ch;
    cudaMalloc(&d_ch, sizeof(char));
    
    // Copy to device
    cudaMemcpy(d_ch, &ch, sizeof(char), cudaMemcpyHostToDevice);
    
    // Launch kernel
    printCharKernel<<<1, 1>>>(d_ch);
    
    cudaDeviceSynchronize();
    cudaFree(d_ch);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void printCharKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Reads from GPU memory and prints |

---

## CUDA Concepts Covered

### 1. Working with char Type in CUDA

CUDA handles `char` type just like `int`, but with 1 byte:

```cuda
char* d_ch;
cudaMalloc(&d_ch, sizeof(char));  // Allocates 1 byte
cudaMemcpy(d_ch, &ch, sizeof(char), cudaMemcpyHostToDevice);
```

### 2. Pointer Dereferencing in Kernel

When passing a single value via pointer, dereference to access it:

```cuda
__global__ void printCharKernel(char* c) {
    printf("%c\n", *c);  // Dereference pointer to get the char value
    //             ↑
    //          *c gives us the actual character
}
```

### 3. scanf with char - Whitespace Handling

The space before `%c` skips any leading whitespace:

```cuda
scanf(" %c", &ch);  // Space before %c skips whitespace/newlines
//     ↑
//  Important: prevents reading leftover newline from previous input
```

### 4. printf in CUDA Kernels

CUDA supports `printf` inside `__global__` functions:

```cuda
__global__ void printCharKernel(char* c) {
    if (idx == 0) {
        printf("%c\n", *c);  // GPU-side printf
    }
}
```

> ⚠️ **Note**: `printf` in kernels requires `cudaDeviceSynchronize()` to ensure output is flushed before program exits.

### 5. Minimal Data Transfer

For a single character, the memory operations are minimal:

```
Host                          Device
┌───┐                         ┌───┐
│'A'│ ──cudaMemcpy (1 byte)──►│'A'│
└───┘                         └───┘
  ch                           d_ch
                                 │
                                 ▼
                          printf("%c", *d_ch)
                                 │
                                 ▼
                            Output: A
```

---

## Alternative Solution

You can also pass the character directly to the kernel and handle output differently:

```cuda
__global__ void printCharKernel(char* data, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        printf("%c", data[idx]);
    }
}

int main() {
    char ch;
    scanf(" %c", &ch);
    
    char* d_ch;
    cudaMalloc(&d_ch, sizeof(char));
    cudaMemcpy(d_ch, &ch, sizeof(char), cudaMemcpyHostToDevice);
    
    printCharKernel<<<1, 1>>>(d_ch, 1);
    printf("\n");  // Newline from host
    
    cudaDeviceSynchronize();
    cudaFree(d_ch);
    
    return 0;
}
```

---

## Common Mistakes

### ❌ Missing Space in scanf
```cuda
scanf("%c", &ch);  // May read leftover newline from input buffer!
```
Always use `scanf(" %c", &ch)` with a leading space for char input.

### ❌ Forgetting to Dereference Pointer
```cuda
__global__ void printCharKernel(char* c) {
    printf("%c\n", c);  // Wrong! This prints the pointer address, not the char
}
```

### ❌ Missing cudaDeviceSynchronize
```cuda
printCharKernel<<<1, 1>>>(d_ch);
// Missing cudaDeviceSynchronize()!
return 0;  // Program may exit before printf output is flushed
```

### ❌ Using %s Instead of %c
```cuda
printf("%s\n", *c);  // Wrong! %s expects a string (char*), not a single char
printf("%c\n", *c);  // Correct! %c for single character
```

### ❌ Passing Host Pointer to Kernel
```cuda
char ch = 'A';
printCharKernel<<<1, 1>>>(&ch);  // Error! Cannot pass host address to kernel
```

---

## Data Types in CUDA

This problem introduces working with `char`. Here's a reference for common types:

| Type | Size | cudaMalloc Example |
|------|------|-------------------|
| `char` | 1 byte | `cudaMalloc(&d_c, sizeof(char))` |
| `int` | 4 bytes | `cudaMalloc(&d_i, sizeof(int))` |
| `float` | 4 bytes | `cudaMalloc(&d_f, sizeof(float))` |
| `double` | 8 bytes | `cudaMalloc(&d_d, sizeof(double))` |
| `long long` | 8 bytes | `cudaMalloc(&d_ll, sizeof(long long))` |

---

## Key Takeaways

1. **char type** works the same as other types in CUDA memory operations
2. **Pointer dereferencing** (`*ptr`) is needed to access values in kernel
3. **scanf whitespace** — use `" %c"` with leading space for char input
4. **printf in kernels** requires `cudaDeviceSynchronize()` to flush output
5. **Any data type** can be transferred between host and device memory

---

## Practice Exercises

1. Modify to read and print a string of N characters
2. Read a character and print its ASCII value
3. Read a lowercase letter and print its uppercase equivalent (and vice versa)
4. Print the character N times using N threads

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/9)*
