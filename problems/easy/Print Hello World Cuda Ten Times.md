# Print Hello World Cuda Ten Times

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Print the phrase **"Hello World Cuda"** exactly ten times. Each occurrence should be printed on its own line.

### Input
There is no input.

### Output
Output the line below exactly ten times, each on a separate line:
```
Hello World Cuda
Hello World Cuda
Hello World Cuda
Hello World Cuda
Hello World Cuda
Hello World Cuda
Hello World Cuda
Hello World Cuda
Hello World Cuda
Hello World Cuda
```

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void helloKernel(int* dummy) {
    int idx = threadIdx.x;
    if (idx < 10) {
        printf("Hello World Cuda\n");
    }
}

int main() {
    int* d_dummy;
    cudaMalloc(&d_dummy, sizeof(int));
    
    // Launch kernel with 10 threads
    helloKernel<<<1, 10>>>(d_dummy);
    
    cudaDeviceSynchronize();
    cudaFree(d_dummy);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void helloKernel()` |
| ✅ Uses parallelism | `threadIdx.x` to identify each thread |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaFree` |
| ✅ Kernel called | `<<<1, 10>>>` launches 10 threads |
| ✅ Meaningful computation | Each thread prints output |

---

## CUDA Concepts Covered

### 1. Launching Multiple Threads

In CUDA, you can launch multiple threads to execute the same kernel code in parallel:

```cuda
helloKernel<<<1, 10>>>(d_dummy);
//           │   │
//           │   └── 10 threads per block
//           └───── 1 block
```

This creates **10 threads**, each executing `helloKernel` simultaneously.

### 2. Thread Identification with `threadIdx`

Each thread has a unique identifier accessible via `threadIdx`:

```
Block 0:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ T0  │ T1  │ T2  │ T3  │ T4  │ T5  │ T6  │ T7  │ T8  │ T9  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  │
  └── threadIdx.x = 0, 1, 2, ..., 9
```

### 3. Bounds Checking

When launching threads, always check bounds to prevent out-of-range operations:

```cuda
if (idx < 10) {
    // Safe to execute
}
```

This is crucial when the number of threads might exceed the required work.

### 4. Parallel Output

Each of the 10 threads executes `printf` independently:

```
Thread 0: printf("Hello World Cuda\n");
Thread 1: printf("Hello World Cuda\n");
Thread 2: printf("Hello World Cuda\n");
...
Thread 9: printf("Hello World Cuda\n");
```

> **Note**: The order of output from parallel threads is **not guaranteed**. However, for this problem, since all outputs are identical, the order doesn't matter.

---

## Common Mistakes

### ❌ Using a Loop Instead of Parallelism
```cuda
// This works but doesn't demonstrate parallelism
__global__ void helloKernel(int* dummy) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < 10; i++) {
            printf("Hello World Cuda\n");
        }
    }
}
```
While this produces correct output, it defeats the purpose of learning parallel programming.

### ❌ Launching Wrong Number of Threads
```cuda
helloKernel<<<1, 5>>>(d_dummy);  // Only 5 outputs!
helloKernel<<<1, 15>>>(d_dummy); // 15 outputs without bounds check!
```

### ❌ Missing Bounds Check
```cuda
__global__ void helloKernel(int* dummy) {
    printf("Hello World Cuda\n");  // All threads print, even extras
}
```

---

## Key Takeaways

1. **Multiple threads** can be launched with `<<<blocks, threads>>>` syntax
2. **`threadIdx.x`** gives each thread a unique identifier within a block
3. **Bounds checking** prevents extra threads from executing unwanted operations
4. **Parallel output** order is non-deterministic, but identical outputs mask this

---

## Practice Exercises

1. Modify to print 100 times using `<<<1, 100>>>`
2. Try using 2 blocks with 5 threads each: `<<<2, 5>>>` and use `blockIdx.x * blockDim.x + threadIdx.x`
3. Print thread numbers: `printf("Hello from thread %d\n", threadIdx.x);`

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/2)*
