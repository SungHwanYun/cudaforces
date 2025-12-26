# 📘 CUDA Online Judge Coding Guide

**User Guide for GPU Programming Learning Platform**

---

## 📚 Table of Contents

1. [System Overview](#1-system-overview)
2. [What is CPU Transpiler?](#2-what-is-cpu-transpiler)
3. [Code Validation](#3-code-validation)
4. [Supported Features](#4-supported-features)
5. [Available Libraries](#5-available-libraries)
6. [Prohibited Items](#6-prohibited-items)
7. [Error Code Reference](#7-error-code-reference)
8. [Coding Guidelines](#8-coding-guidelines)
9. [Notes and Limitations](#9-notes-and-limitations)
10. [Frequently Asked Questions (FAQ)](#10-frequently-asked-questions-faq)

---

## 1. System Overview

### 1.1 What is CUDA Online Judge?

CUDA Online Judge is an educational online judging system for learning GPU programming. 
You can write and test CUDA code even without an actual GPU.

### 1.2 Judging Process

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  CUDA Code  │ ---> │ Transpiler  │ ---> │  C++ Code   │ ---> │ CPU Execute │
│  (Submit)   │ Conv │             │ Gen  │             │ Comp │  & Judge    │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
```

- Submitted CUDA code is **converted to CPU-executable C++ code**
- OpenMP is used to simulate GPU parallel processing
- Only output **correctness** is evaluated

### 1.3 Judgment Results

| Result | Description |
|--------|-------------|
| ✅ **Accepted (AC)** | Correct - All test cases passed |
| ❌ **Wrong Answer (WA)** | Incorrect - Output differs from expected |
| ⚠️ **Compile Error (CE)** | Compilation error - Syntax error |
| ⏱️ **Time Limit Exceeded (TLE)** | Time limit exceeded |
| 💾 **Memory Limit Exceeded (MLE)** | Memory limit exceeded |
| 🚫 **Runtime Error (RE)** | Runtime error - Segfault, etc. |
| 🔒 **Validation Error (VE)** | Validation failed - CUDA rule violation |

---

## 2. What is CPU Transpiler?

### 2.1 Concept

**CPU Transpiler** is a system that converts CUDA code to run on CPU without a GPU.

```cuda
// Original CUDA code
__global__ void add(int* a, int* b, int* c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Kernel launch
add<<<1, 5>>>(d_a, d_b, d_c);
```

The above code is converted to:

```cpp
// Converted C++ code (OpenMP parallelization)
void add_impl(int threadIdx_x, ..., int* a, int* b, int* c) {
    struct { int x; } threadIdx = {threadIdx_x};
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Kernel launch → OpenMP loop
#pragma omp parallel for
for (int tx = 0; tx < 5; tx++) {
    add_impl(tx, ..., d_a, d_b, d_c);
}
```

### 2.2 Conversion Method

| CUDA Element | CPU Conversion |
|--------------|----------------|
| `__global__` function | Regular C++ function |
| `<<<blocks, threads>>>` | OpenMP nested loops |
| `threadIdx.x/y/z` | Function parameters |
| `blockIdx.x/y/z` | Function parameters |
| `blockDim.x/y/z` | Function parameters |
| `cudaMalloc` | `malloc` |
| `cudaMemcpy` | `memcpy` |
| `cudaFree` | `free` |
| `__shared__` | Per-block independent array |
| `atomicAdd` | `__atomic_fetch_add` |
| `__syncthreads()` | `#pragma omp barrier` |

### 2.3 ⚠️ Performance Benchmarking Not Available

> **Important**: This system is intended for **correctness verification** only.

**Why performance measurement is meaningless:**

1. **Different execution environment from actual GPU**
   - GPU executes thousands of threads simultaneously
   - CPU emulation simulates sequentially
   
2. **Memory structure difference**
   - No GPU high-speed memory hierarchy (L1/L2/Shared/Global)
   - All memory mapped to system RAM

3. **Time complexity difference**
   - O(1) parallel operations on GPU may become O(n) on CPU

```cuda
// GPU performance of this code ≠ CPU transpiled performance
__global__ void matMul(float* A, float* B, float* C, int N) {
    // GPU: All threads execute simultaneously
    // CPU: Threads simulated sequentially
}
```

**Recommendations:**
- Only verify algorithm **correctness**
- Test actual performance optimization in GPU environment
- Time measured with `cudaEvent` is meaningless (always returns 0)

---

## 3. Code Validation

Submitted code must pass the following validations:

### 3.1 Required Criteria

| Validation Item | Description | Error Code |
|-----------------|-------------|------------|
| ✅ **Kernel exists** | At least one `__global__` function required | E3001 |
| ✅ **Meaningful computation** | Kernel must perform actual computation | E3002 |
| ✅ **Uses parallelism** | Must use `threadIdx`, `blockIdx`, etc. | E3003 |
| ✅ **Uses GPU memory** | Must use `cudaMalloc`, `cudaMemcpy` | E3004 |
| ✅ **Kernel called** | Must call defined kernel with `<<<>>>` syntax | E3001 |
| ❌ **No forbidden functions** | Cannot use `qsort`, STL functions, etc. | E3005 |
| ❌ **No forbidden types** | Cannot use `std::vector`, `std::string`, etc. | E3006 |

### 3.2 Meaningful Kernel Conditions

Must satisfy **at least one** of the following:

```cuda
// ✅ Condition 1: Performs computation
__global__ void kernel1(int* a, int* b, int* c) {
    c[i] = a[i] + b[i];  // Arithmetic operation
}

// ✅ Condition 2: Uses parameters
__global__ void kernel2(int* data, int n) {
    data[threadIdx.x] = n;  // Parameter access
}

// ✅ Condition 3: Array access
__global__ void kernel3(int* arr) {
    arr[threadIdx.x] = 1;  // Memory access
}

// ✅ Condition 4: Performs output
__global__ void kernel4() {
    printf("Hello from GPU!\n");  // Output
}
```

### 3.3 Parallelism Validation

Must use **at least one** of these built-in variables inside the kernel:

- `threadIdx.x`, `threadIdx.y`, `threadIdx.z`
- `blockIdx.x`, `blockIdx.y`, `blockIdx.z`
- `blockDim.x`, `blockDim.y`, `blockDim.z`
- `gridDim.x`, `gridDim.y`, `gridDim.z`

```cuda
// ❌ Validation failed: No parallelism used
__global__ void bad_kernel(int* arr) {
    arr[0] = 1;  // All threads do the same work
}

// ✅ Validation passed: Parallelism used
__global__ void good_kernel(int* arr) {
    int i = threadIdx.x;  // Different index per thread
    arr[i] = i;
}
```

### 3.4 GPU Memory Validation

Must use **at least one** of these functions in the main function:

- `cudaMalloc`
- `cudaMemcpy` (or `cudaMemcpyAsync`)
- `cudaMemset` (or `cudaMemsetAsync`)

```cuda
// ❌ Validation failed: No GPU memory used
int main() {
    int arr[10];
    kernel<<<1, 10>>>(arr);  // Directly passing host memory
}

// ✅ Validation passed: GPU memory used
int main() {
    int *d_arr;
    cudaMalloc(&d_arr, 10 * sizeof(int));  // GPU memory allocation
    cudaMemcpy(d_arr, arr, 10 * sizeof(int), cudaMemcpyHostToDevice);
    kernel<<<1, 10>>>(d_arr);
    cudaFree(d_arr);
}
```

---

## 4. Supported Features

### 4.1 Function Keywords

| Keyword | Description | Support |
|---------|-------------|---------|
| `__global__` | Kernel function running on GPU | ✅ |
| `__device__` | Function callable only from GPU | ✅ |
| `__host__` | Function running on CPU | ✅ |
| `__host__ __device__` | Callable from both CPU/GPU | ✅ |

### 4.2 Memory Keywords

| Keyword | Description | Support |
|---------|-------------|---------|
| `__shared__` | Block shared memory (static) | ✅ |
| `extern __shared__` | Dynamic shared memory | ✅ |
| `__device__` | Device global variable | ✅ |
| `__constant__` | Constant memory | ✅ |

### 4.3 Built-in Variables

| Variable | Description | Support |
|----------|-------------|---------|
| `threadIdx.x/y/z` | Thread index within block | ✅ |
| `blockIdx.x/y/z` | Block index within grid | ✅ |
| `blockDim.x/y/z` | Threads per block | ✅ |
| `gridDim.x/y/z` | Blocks in grid | ✅ |
| `warpSize` | Warp size (32) | ✅ |

### 4.4 Memory Management Functions

| Function | Description | Support |
|----------|-------------|---------|
| `cudaMalloc` | GPU memory allocation | ✅ |
| `cudaFree` | GPU memory deallocation | ✅ |
| `cudaMemcpy` | Memory copy | ✅ |
| `cudaMemcpyAsync` | Async memory copy | ✅ |
| `cudaMemset` | Memory initialization | ✅ |
| `cudaMemsetAsync` | Async memory initialization | ✅ |
| `cudaMemcpyToSymbol` | Copy to symbol | ✅ |
| `cudaMemcpyFromSymbol` | Copy from symbol | ✅ |
| `cudaMemGetInfo` | Memory info query | ✅ |

### 4.5 Atomic Operations

| Function | Description | Support |
|----------|-------------|---------|
| `atomicAdd` | Atomic addition | ✅ |
| `atomicSub` | Atomic subtraction | ✅ |
| `atomicExch` | Atomic exchange | ✅ |
| `atomicMin` | Atomic minimum | ✅ |
| `atomicMax` | Atomic maximum | ✅ |
| `atomicInc` | Atomic increment (modular) | ✅ |
| `atomicDec` | Atomic decrement (modular) | ✅ |
| `atomicCAS` | Compare-And-Swap | ✅ |
| `atomicAnd` | Atomic AND | ✅ |
| `atomicOr` | Atomic OR | ✅ |
| `atomicXor` | Atomic XOR | ✅ |

### 4.6 Synchronization Functions

| Function | Description | Support |
|----------|-------------|---------|
| `__syncthreads()` | Block thread synchronization | ✅ |
| `__syncwarp()` | Warp synchronization | ✅ |
| `cudaDeviceSynchronize()` | Device synchronization | ✅ |
| `cudaStreamSynchronize()` | Stream synchronization | ✅ |

### 4.7 Warp Operations

| Function | Description | Support |
|----------|-------------|---------|
| `__shfl_sync` | Warp shuffle | ✅ |
| `__shfl_up_sync` | Up shuffle | ✅ |
| `__shfl_down_sync` | Down shuffle | ✅ |
| `__shfl_xor_sync` | XOR shuffle | ✅ |
| `__ballot_sync` | Warp vote | ✅ |
| `__all_sync` | All true check | ✅ |
| `__any_sync` | Any true check | ✅ |
| `__activemask()` | Active thread mask | ✅ |

### 4.8 Streams and Events

| Function | Description | Support |
|----------|-------------|---------|
| `cudaStreamCreate` | Create stream | ✅ |
| `cudaStreamDestroy` | Destroy stream | ✅ |
| `cudaStreamSynchronize` | Synchronize stream | ✅ |
| `cudaEventCreate` | Create event | ✅ |
| `cudaEventRecord` | Record event | ✅ |
| `cudaEventSynchronize` | Synchronize event | ✅ |
| `cudaEventElapsedTime` | Elapsed time (always 0) | ⚠️ |

### 4.9 Texture Memory

| Function | Description | Support |
|----------|-------------|---------|
| `tex1D` | 1D texture read | ✅ |
| `tex2D` | 2D texture read | ✅ |
| `tex1Dfetch` | 1D integer coordinate | ✅ |
| `tex2Dfetch` | 2D integer coordinate | ✅ |

### 4.10 Data Types

| Type | Support |
|------|---------|
| `int`, `unsigned int` | ✅ |
| `float`, `double` | ✅ |
| `char`, `unsigned char` | ✅ |
| `short`, `unsigned short` | ✅ |
| `long`, `unsigned long` | ✅ |
| `long long`, `unsigned long long` | ✅ |
| `size_t` | ✅ |
| `bool` | ✅ |
| `void` | ✅ |
| `dim3` | ✅ |
| Pointers (`int*`, `float**`) | ✅ |
| Arrays (`int arr[N]`) | ✅ |
| Multi-dimensional arrays | ✅ |
| `struct` | ✅ |
| `enum` | ✅ |
| `typedef` | ✅ |

### 4.11 Operators

| Category | Operators | Support |
|----------|-----------|---------|
| Arithmetic | `+`, `-`, `*`, `/`, `%` | ✅ |
| Comparison | `==`, `!=`, `<`, `>`, `<=`, `>=` | ✅ |
| Logical | `&&`, `\|\|`, `!` | ✅ |
| Bitwise | `&`, `\|`, `^`, `~`, `<<`, `>>` | ✅ |
| Assignment | `=`, `+=`, `-=`, `*=`, `/=`, `%=` | ✅ |
| Assignment (bitwise) | `&=`, `\|=`, `^=`, `<<=`, `>>=` | ✅ |
| Increment/Decrement | `++`, `--` (prefix/postfix) | ✅ |
| Ternary | `? :` | ✅ |
| Pointer | `*`, `&`, `->` | ✅ |
| sizeof | `sizeof(type)`, `sizeof(expr)` | ✅ |
| Casting | `(type)expr` | ✅ |

### 4.12 Control Structures

| Structure | Support |
|-----------|---------|
| `if-else` | ✅ |
| `for` loop | ✅ |
| `while` loop | ✅ |
| `do-while` loop | ✅ |
| `switch-case-default` | ✅ |
| `break` | ✅ |
| `continue` | ✅ |
| `return` | ✅ |

---

## 5. Available Libraries

### 5.1 Allowed Headers (stdio.h)

```c
// I/O functions
printf()     // Output
scanf()      // Input
fprintf()    // File output
fscanf()     // File input
fopen()      // Open file
fclose()     // Close file
fread()      // Binary read
fwrite()     // Binary write
fgets()      // Line read
fputs()      // Line write
getchar()    // Character input
putchar()    // Character output
```

### 5.2 Allowed Functions (stdlib.h partial)

```c
// Memory management
malloc()     // Memory allocation
calloc()     // Zero-initialized allocation
realloc()    // Memory reallocation
free()       // Memory deallocation

// Conversion functions
atoi()       // String → integer
atof()       // String → float
atol()       // String → long
strtol()     // String → long (with base)
strtod()     // String → double

// Random
rand()       // Random number
srand()      // Seed setting

// Misc
abs()        // Absolute value
exit()       // Program exit
```

### 5.3 Allowed Math Functions (math.h)

```c
// Basic math functions
sin(), cos(), tan()      // Trigonometric
asin(), acos(), atan()   // Inverse trigonometric
sinh(), cosh(), tanh()   // Hyperbolic
exp(), log(), log10()    // Exponential/logarithm
pow(), sqrt()            // Power/square root
ceil(), floor(), round() // Ceiling/floor/round
fabs(), fmod()           // Absolute value/modulo
fmin(), fmax()           // Minimum/maximum
```

### 5.4 Allowed Functions (string.h / cstring)

```c
// String copy/concatenate
strcpy()     // String copy
strncpy()    // String copy (n characters)
strcat()     // String concatenate
strncat()    // String concatenate (n characters)

// String comparison
strcmp()     // String compare
strncmp()    // String compare (n characters)

// String search
strlen()     // String length
strchr()     // Find character (first)
strrchr()    // Find character (last)
strstr()     // Find substring
strpbrk()    // Find any character from set
strspn()     // Span of characters in set
strcspn()    // Span of characters not in set
strtok()     // Tokenize string

// Memory manipulation
memcpy()     // Memory copy
memmove()    // Memory move (overlap safe)
memcmp()     // Memory compare
memset()     // Memory set
memchr()     // Find byte in memory
```

---

## 6. Prohibited Items

### 6.1 Prohibited Function List

> ⚠️ The following functions must be **implemented yourself**.

#### stdlib.h Prohibited Functions
```c
qsort()      // ❌ Must implement sorting yourself
bsearch()    // ❌ Must implement binary search yourself
```

#### STL algorithm Functions
```c
// Sorting
sort(), stable_sort(), partial_sort(), nth_element()

// Searching
find(), find_if(), find_first_of(), binary_search()
lower_bound(), upper_bound(), equal_range()

// Modification
copy(), fill(), transform(), replace(), swap()
reverse(), rotate(), shuffle(), unique(), remove()

// Aggregation
count(), count_if(), accumulate(), inner_product()
min(), max(), min_element(), max_element()

// Iteration
for_each(), all_of(), any_of(), none_of()
```

#### STL Container Methods
```c
push_back(), pop_back(), push_front(), pop_front()
emplace(), insert(), erase(), clear(), resize()
begin(), end(), front(), back(), at(), size()
```

### 6.2 Prohibited Type List

> ⚠️ C++ STL containers cannot be used. Use **C-style arrays and pointers**.

```cpp
// ❌ Prohibited STL containers
std::vector<T>         // → Use int arr[N] or int* arr
std::string            // → Use char arr[N] or char*
std::map<K,V>          // → Must implement yourself
std::unordered_map<K,V>
std::set<T>
std::unordered_set<T>
std::list<T>
std::deque<T>
std::queue<T>
std::stack<T>
std::priority_queue<T>
std::pair<T,U>
std::tuple<...>
std::array<T,N>
std::bitset<N>

// ❌ Prohibited synchronization types
std::mutex
std::thread
std::atomic<T>

// ❌ Prohibited smart pointers
std::shared_ptr<T>
std::unique_ptr<T>
std::weak_ptr<T>
```

### 6.3 Correct Alternatives

```cuda
// ❌ Wrong (using vector)
std::vector<int> arr(N);

// ✅ Correct (C-style array)
int arr[N];           // Fixed size
int* arr = (int*)malloc(N * sizeof(int));  // Dynamic allocation

// ❌ Wrong (using string)
std::string str = "hello";

// ✅ Correct (C-style string)
char str[] = "hello";
char* str = "hello";

// ❌ Wrong (using sort)
std::sort(arr, arr + n);

// ✅ Correct (implement yourself)
__device__ void bubbleSort(int* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}
```

---

## 7. Error Code Reference

### 7.1 E1xxx: Syntax Error (SYNTAX_ERROR)

> CUDA C syntax errors that users must fix

| Code | Name | Description | Solution |
|------|------|-------------|----------|
| E1001 | UNEXPECTED_TOKEN | Unexpected token | Check syntax |
| E1002 | MISSING_SEMICOLON | Missing semicolon | Add `;` |
| E1003 | UNMATCHED_BRACKET | Bracket mismatch | Check `{}`, `()`, `[]` pairs |
| E1004 | INVALID_EXPRESSION | Invalid expression | Check expression syntax |
| E1005 | MISSING_TYPE | Missing type | Specify variable/function type |
| E1006 | INVALID_DECLARATION | Invalid declaration | Check declaration statement |
| E1007 | EXPECTED_IDENTIFIER | Identifier expected | Check variable/function name |
| E1008 | INVALID_KERNEL_LAUNCH | Invalid kernel launch | Check `<<<>>>` syntax |
| E1009 | INVALID_ARRAY_SIZE | Invalid array size | Check array size |
| E1010 | INVALID_OPERATOR | Invalid operator | Check operator |

### 7.2 E2xxx: Semantic Error (SEMANTIC_ERROR)

> CUDA semantic rule violations

| Code | Name | Description | Solution |
|------|------|-------------|----------|
| E2001 | INVALID_MEMCPY_DIRECTION | cudaMemcpy direction error | Check Host/Device pointers |
| E2002 | DEVICE_VAR_IN_HOST | Accessing __device__ from host | Use cudaMemcpyToSymbol |
| E2003 | HOST_VAR_IN_DEVICE | Accessing host variable in kernel | Pass as parameter |
| E2004 | SHARED_MEMORY_IN_HOST | Using __shared__ in host | Use only inside kernel |
| E2005 | CONSTANT_WRITE_IN_KERNEL | Writing __constant__ in kernel | __constant__ is read-only |
| E2006 | INVALID_MEMORY_ACCESS | Invalid memory access | Check pointers |
| E2007 | HOST_FUNC_IN_KERNEL | Calling host function in kernel | Use __device__ function |
| E2008 | HOST_VAR_ACCESS_IN_KERNEL | Accessing host variable in kernel | Use __device__ or parameter |
| E2009 | DEVICE_FUNC_IN_HOST | Calling __device__ from host | Add __host__ __device__ |
| E2010 | GLOBAL_FUNC_IN_KERNEL | Calling kernel from kernel | Dynamic parallelism not supported |

### 7.3 E3xxx: Validation Error (VALIDATION_ERROR)

> OJ policy violations - Educational purpose restrictions

| Code | Name | Description | Solution |
|------|------|-------------|----------|
| E3001 | NO_KERNEL_FOUND | No kernel function / not called | Write and call `__global__` function |
| E3002 | KERNEL_NOT_SIGNIFICANT | Kernel is meaningless | Add code that performs actual computation |
| E3003 | NO_PARALLELISM | No parallelism used | Use `threadIdx`, `blockIdx` |
| E3004 | NO_GPU_MEMORY_OPS | No GPU memory used | Use `cudaMalloc`, `cudaMemcpy` |
| E3005 | FORBIDDEN_FUNCTION | Forbidden function used | See [Prohibited Function List](#61-prohibited-function-list) |
| E3006 | FORBIDDEN_TYPE | Forbidden type used | See [Prohibited Type List](#62-prohibited-type-list) |

### 7.4 E4xxx: Not Supported (NOT_SUPPORTED)

> CUDA features not supported by transpiler

| Code | Name | Description | Alternative |
|------|------|-------------|-------------|
| E4001 | UNSUPPORTED_FEATURE | General unsupported feature | See documentation |
| E4002 | COMPLEX_TEMPLATE | Complex templates | Simplification needed |
| E4003 | INLINE_PTX | Inline PTX assembly | Replace with C++ code |
| E4005 | DYNAMIC_PARALLELISM | Dynamic parallelism | Call kernel from host |
| E4006 | COOPERATIVE_GROUPS | Cooperative groups | Use __syncthreads |
| E4008 | UNIFIED_MEMORY | Unified memory | cudaMalloc + cudaMemcpy |

### 7.5 E5xxx: Internal Error (INTERNAL_ERROR)

> System internal errors (not user's responsibility)

| Code | Name | Description |
|------|------|-------------|
| E5001 | PARSER_INTERNAL | Parser internal error |
| E5002 | TRANSPILER_INTERNAL | Transpiler internal error |
| E5003 | CODE_GEN_FAILED | Code generation failed |
| E5999 | UNKNOWN_INTERNAL | Unknown internal error |

---

## 8. Coding Guidelines

### 8.1 Basic Template

```cuda
#include <stdio.h>

// Kernel function definition
__global__ void myKernel(int* input, int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Perform actual computation
        output[idx] = input[idx] * 2;
    }
}

int main() {
    int n;
    scanf("%d", &n);
    
    // Host memory allocation
    int* h_input = (int*)malloc(n * sizeof(int));
    int* h_output = (int*)malloc(n * sizeof(int));
    
    // Read input
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_input[i]);
    }
    
    // Device memory allocation
    int *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    
    // Host → Device copy
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Kernel execution
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    myKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    
    // Device → Host copy
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    for (int i = 0; i < n; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");
    
    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}
```

### 8.2 Atomic Operation Example

```cuda
#include <stdio.h>

__global__ void sumKernel(int* arr, int n, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        atomicAdd(result, arr[idx]);  // Atomic addition
    }
}

int main() {
    int n = 1000;
    int* h_arr = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) h_arr[i] = 1;
    
    int *d_arr, *d_result;
    int h_result = 0;
    
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);
    
    sumKernel<<<10, 100>>>(d_arr, n, d_result);
    
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Sum: %d\n", h_result);  // 1000
    
    cudaFree(d_arr);
    cudaFree(d_result);
    free(h_arr);
    
    return 0;
}
```

### 8.3 Shared Memory Example

```cuda
#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void sharedMemSum(int* input, int* output, int n) {
    __shared__ int shared_data[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    shared_data[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Store block result
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}
```

### 8.4 2D Grid/Block Example

```cuda
#include <stdio.h>

#define N 16
#define BLOCK_SIZE 4

__global__ void matrixAdd(int* A, int* B, int* C, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < width && col < width) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int size = N * N * sizeof(int);
    
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;
    
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);
    
    // Initialize
    for (int i = 0; i < N * N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // 2D grid setup
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matrixAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Print results (first row only)
    for (int i = 0; i < N; i++) {
        printf("%d ", h_C[i]);
    }
    printf("\n");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
```

---

## 9. Notes and Limitations

### 9.1 Behavioral Differences

| Item | GPU Behavior | CPU Emulation |
|------|--------------|---------------|
| Parallel execution | Simultaneous | Sequential simulation |
| `cudaEvent` time | Actual measurement | Always 0 |
| Memory bandwidth | High speed | System RAM speed |
| Warp synchronization | Hardware support | Software emulation |
| Streams | Asynchronous | Synchronous |

### 9.2 Best Practices

1. **Always perform bounds checking**
   ```cuda
   if (idx < n) {
       // Safe access
   }
   ```

2. **Don't forget to free memory**
   ```cuda
   cudaFree(d_ptr);
   free(h_ptr);
   ```

3. **Use appropriate block sizes**
   ```cuda
   // Generally 128 ~ 512 recommended
   int blockSize = 256;
   ```

4. **Consider shared memory size limits**
   ```cuda
   // 48KB per block limit (actual GPU)
   __shared__ float data[1024];  // 4KB
   ```

5. **Prevent race conditions**
   ```cuda
   atomicAdd(&sum, value);  // Use atomic for concurrent writes
   ```

---

## 10. Frequently Asked Questions (FAQ)

### Q1: Why can't I use `std::vector`?

**A:** For educational purposes, it's intended for you to learn C-style memory management.
In actual CUDA development, GPU memory is also managed with C-style pointers.

```cuda
// ❌ Prohibited
std::vector<int> arr(N);

// ✅ Recommended
int* arr = (int*)malloc(N * sizeof(int));
```

### Q2: Execution time seems slow. Is my code inefficient?

**A:** No. The CPU transpiler is for **correctness verification**, so performance differs from actual GPU.
Actual performance testing should be done in a GPU environment.

### Q3: `cudaEventElapsedTime` always returns 0.

**A:** This is normal. In CPU emulation, all operations are executed synchronously,
so elapsed time measurement is meaningless.

### Q4: I'm getting E3003 (NO_PARALLELISM) error.

**A:** You must use parallelism variables like `threadIdx`, `blockIdx` inside the kernel.

```cuda
// ❌ Error occurs
__global__ void kernel(int* arr) {
    arr[0] = 1;  // All threads do same work
}

// ✅ Solution
__global__ void kernel(int* arr) {
    int i = threadIdx.x;  // Different index per thread
    arr[i] = i;
}
```

### Q5: How do I use dynamic shared memory?

**A:** Use `extern __shared__` and the third argument in kernel launch.

```cuda
extern __shared__ int shared[];

__global__ void kernel(int* data, int n) {
    int tid = threadIdx.x;
    shared[tid] = data[tid];
    __syncthreads();
    // ...
}

int main() {
    // Third argument: dynamic shared memory size (bytes)
    kernel<<<1, 256, 256 * sizeof(int)>>>(d_data, n);
}
```

### Q6: Why am I getting cudaMemcpy direction error (E2001)?

**A:** The direction doesn't match between pointers allocated with `cudaMalloc` and host pointers.

```cuda
// ❌ Error: d_arr is Device but used as src
cudaMemcpy(h_arr, d_arr, size, cudaMemcpyHostToDevice);

// ✅ Correct: d_arr is src, DeviceToHost
cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
```

---

## 📞 Help and Support

- **Contact**: 📧 [ejpark29@gmail.com](mailto:ejpark29@gmail.com)

---

**Version**: 2.1.0  
**Last Updated**: December 2025

---

**Happy CUDA Learning! 🚀**
