# Echo Integer

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given an integer a. Print the integer a as is.

### Input
The first line contains an integer a.

It is guaranteed that 1 ≤ a ≤ 100.

### Output
Print the integer a on the first line.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 10 | 10 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 42 | 42 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 99 | 99 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void echoIntKernel(int* value) {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("%d\n", *value);
    }
}

int main() {
    int a;
    scanf("%d", &a);
    
    // Device memory
    int* d_value;
    cudaMalloc(&d_value, sizeof(int));
    
    // Copy to device
    cudaMemcpy(d_value, &a, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    echoIntKernel<<<1, 1>>>(d_value);
    
    cudaDeviceSynchronize();
    cudaFree(d_value);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void echoIntKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Reads from GPU memory and prints |

---

## CUDA Concepts Covered

### 1. Basic Integer Transfer

Transferring a single integer to GPU memory:

```cuda
int a;
scanf("%d", &a);

int* d_value;
cudaMalloc(&d_value, sizeof(int));  // 4 bytes for int
cudaMemcpy(d_value, &a, sizeof(int), cudaMemcpyHostToDevice);
```

### 2. Pointer Dereferencing in Kernel

Access the value through pointer dereferencing:

```cuda
__global__ void echoIntKernel(int* value) {
    printf("%d\n", *value);  // Dereference to get the int value
    //             ↑
    //          *value gives us the actual integer
}
```

### 3. Memory Flow Visualization

```
┌─────────────────────────────────────────────────────────┐
│                     HOST (CPU)                          │
│  1. scanf("%d", &a)        ← Read input (a = 42)        │
│  2. cudaMalloc(&d_value, 4)← Allocate 4 bytes on GPU    │
│  3. cudaMemcpy(...)        ← Copy value to GPU          │
│  4. echoIntKernel<<<1,1>>> ← Launch kernel              │
│  5. cudaDeviceSynchronize()← Wait for completion        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    DEVICE (GPU)                         │
│  d_value points to: [42]                                │
│  printf("%d\n", *d_value) → Output: 42                  │
└─────────────────────────────────────────────────────────┘
```

### 4. Comparing String vs Integer Echo

| Aspect | Echo String | Echo Integer |
|--------|-------------|--------------|
| Data type | `char*` | `int*` |
| Size | `len + 1` bytes | 4 bytes |
| Format | `%s` or `%c` | `%d` |
| Terminator | Needs `'\0'` | Not needed |

### 5. Why Use GPU for Simple Output?

This problem demonstrates the **minimal CUDA pattern**:

```
Read → Allocate → Copy → Execute → Sync → Free
```

Even simple operations follow this pattern to satisfy CUDA validation requirements.

---

## Alternative Solutions

### Passing Value as Kernel Parameter

```cuda
__global__ void echoIntKernel(int value) {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("%d\n", value);  // Direct value, no pointer needed
    }
}

int main() {
    int a;
    scanf("%d", &a);
    
    // Still need GPU memory for validation
    int* d_dummy;
    cudaMalloc(&d_dummy, sizeof(int));
    
    // Pass value directly (copied by value)
    echoIntKernel<<<1, 1>>>(a);
    
    cudaDeviceSynchronize();
    cudaFree(d_dummy);
    
    return 0;
}
```

> ⚠️ **Note**: We still allocate GPU memory to pass validation, even though the value is passed directly.

### Using Array Style

```cuda
__global__ void echoIntKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("%d\n", data[0]);  // Array access instead of pointer dereference
    }
}
```

Both `*data` and `data[0]` are equivalent for accessing the first element.

---

## Common Mistakes

### ❌ Forgetting to Dereference
```cuda
__global__ void echoIntKernel(int* value) {
    printf("%d\n", value);  // Wrong! This prints the pointer address
}
```
Use `*value` to get the actual integer.

### ❌ Wrong Format Specifier
```cuda
printf("%s\n", *value);  // Wrong! %s is for strings
printf("%c\n", *value);  // Wrong! %c is for characters
printf("%d\n", *value);  // Correct! %d is for integers
```

### ❌ Missing cudaDeviceSynchronize
```cuda
echoIntKernel<<<1, 1>>>(d_value);
return 0;  // May exit before printf completes!
```
Always synchronize before exiting.

### ❌ Passing Host Pointer to Kernel
```cuda
int a = 42;
echoIntKernel<<<1, 1>>>(&a);  // Error! Cannot use host address in kernel
```

---

## Printf Format Specifiers Reference

| Specifier | Type | Example |
|-----------|------|---------|
| `%d` | int | `42` |
| `%ld` | long | `42L` |
| `%lld` | long long | `42LL` |
| `%u` | unsigned int | `42u` |
| `%f` | float/double | `3.14` |
| `%e` | scientific | `3.14e+00` |
| `%c` | char | `'A'` |
| `%s` | string | `"hello"` |
| `%x` | hex | `0x2A` |

---

## Data Type Sizes

| Type | Size | cudaMalloc Example |
|------|------|-------------------|
| `char` | 1 byte | `cudaMalloc(&ptr, sizeof(char))` |
| `int` | 4 bytes | `cudaMalloc(&ptr, sizeof(int))` |
| `float` | 4 bytes | `cudaMalloc(&ptr, sizeof(float))` |
| `double` | 8 bytes | `cudaMalloc(&ptr, sizeof(double))` |
| `long long` | 8 bytes | `cudaMalloc(&ptr, sizeof(long long))` |

---

## Key Takeaways

1. **Single integer** requires 4 bytes of GPU memory
2. **Pointer dereference** (`*ptr`) accesses the value
3. **`%d` format** for integer output
4. **Minimal pattern** — even simple tasks need full CUDA workflow
5. **cudaDeviceSynchronize** ensures printf completes before exit

---

## Practice Exercises

1. Read two integers and print their **sum**
2. Read an integer and print its **square**
3. Read an integer and print whether it's **even or odd**
4. Echo a **long long** integer (adjust format specifier)

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/19)*
