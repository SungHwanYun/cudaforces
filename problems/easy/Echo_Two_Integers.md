# Echo Two Integers

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

You are given two integers a and b separated by a space. Print the integers a and b separated by a space on the same line.

### Input
The first line contains two space-separated integers a and b.

**Constraints:**
- 1 ≤ a, b ≤ 100

### Output
Print the integers a and b separated by a space on the first line.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 1 2 | 1 2 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 42 99 | 42 99 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 100 1 | 100 1 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void echoTwoIntsKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("%d %d\n", data[0], data[1]);
    }
}

int main() {
    int a, b;
    scanf("%d %d", &a, &b);
    
    // Host array
    int h_data[2] = {a, b};
    
    // Device memory
    int* d_data;
    cudaMalloc(&d_data, 2 * sizeof(int));
    
    // Copy to device
    cudaMemcpy(d_data, h_data, 2 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    echoTwoIntsKernel<<<1, 1>>>(d_data);
    
    cudaDeviceSynchronize();
    cudaFree(d_data);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void echoTwoIntsKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Reads from GPU memory array and prints |

---

## CUDA Concepts Covered

### 1. Storing Multiple Values in Array

When handling multiple related values, use an array:

```cuda
int h_data[2] = {a, b};  // Pack two integers into array
//    [0] = a
//    [1] = b
```

### 2. Array Memory Allocation

Allocate space for multiple elements:

```cuda
cudaMalloc(&d_data, 2 * sizeof(int));  // 2 × 4 = 8 bytes
//                   ↑
//         Number of elements × size per element
```

### 3. Array Access in Kernel

Access elements using array indexing:

```cuda
__global__ void echoTwoIntsKernel(int* data) {
    printf("%d %d\n", data[0], data[1]);
    //                  ↑        ↑
    //               first    second
}
```

### 4. Memory Layout

```
Host Array:
┌─────────┬─────────┐
│    a    │    b    │
└─────────┴─────────┘
   [0]       [1]
   
   cudaMemcpy (8 bytes)
         ↓
         
Device Array:
┌─────────┬─────────┐
│    a    │    b    │
└─────────┴─────────┘
   [0]       [1]
```

### 5. Printf with Multiple Values

Output multiple values in a single printf:

```cuda
printf("%d %d\n", data[0], data[1]);
//      ↑  ↑       ↑        ↑
//     format    first   second
//   specifiers  value    value
```

---

## Alternative Solutions

### Using Two Separate Pointers

```cuda
__global__ void echoTwoIntsKernel(int* a, int* b) {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("%d %d\n", *a, *b);
    }
}

int main() {
    int a, b;
    scanf("%d %d", &a, &b);
    
    int *d_a, *d_b;
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
    
    echoTwoIntsKernel<<<1, 1>>>(d_a, d_b);
    
    cudaDeviceSynchronize();
    cudaFree(d_a);
    cudaFree(d_b);
    
    return 0;
}
```

> 💡 **Tip**: Using a single array is cleaner and requires fewer memory operations.

### Passing Values Directly (with Dummy Allocation)

```cuda
__global__ void echoTwoIntsKernel(int a, int b, int* dummy) {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("%d %d\n", a, b);
    }
}

int main() {
    int a, b;
    scanf("%d %d", &a, &b);
    
    int* d_dummy;
    cudaMalloc(&d_dummy, sizeof(int));
    
    echoTwoIntsKernel<<<1, 1>>>(a, b, d_dummy);
    
    cudaDeviceSynchronize();
    cudaFree(d_dummy);
    
    return 0;
}
```

---

## Comparison: One Integer vs Two Integers

| Aspect | Echo Integer | Echo Two Integers |
|--------|--------------|-------------------|
| Memory size | `sizeof(int)` (4 bytes) | `2 * sizeof(int)` (8 bytes) |
| Access | `*data` or `data[0]` | `data[0]`, `data[1]` |
| Printf | `printf("%d", *data)` | `printf("%d %d", data[0], data[1])` |

---

## Common Mistakes

### ❌ Insufficient Memory Allocation
```cuda
cudaMalloc(&d_data, sizeof(int));  // Wrong! Only 4 bytes for 2 integers
cudaMalloc(&d_data, 2 * sizeof(int));  // Correct! 8 bytes
```

### ❌ Wrong Array Indices
```cuda
printf("%d %d\n", data[1], data[2]);  // Wrong! data[2] is out of bounds
printf("%d %d\n", data[0], data[1]);  // Correct!
```

### ❌ Forgetting Space in Output
```cuda
printf("%d%d\n", data[0], data[1]);   // Wrong! Output: "12" not "1 2"
printf("%d %d\n", data[0], data[1]);  // Correct! Output: "1 2"
```

### ❌ Copying Wrong Size
```cuda
cudaMemcpy(d_data, h_data, sizeof(int), ...);  // Wrong! Only copies first int
cudaMemcpy(d_data, h_data, 2 * sizeof(int), ...);  // Correct!
```

---

## Scaling to N Integers

The pattern extends naturally to N values:

```cuda
// For N integers
int h_data[N];
for (int i = 0; i < N; i++) {
    scanf("%d", &h_data[i]);
}

int* d_data;
cudaMalloc(&d_data, N * sizeof(int));
cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
```

---

## Memory Size Formula

```
Total bytes = Number of elements × sizeof(element type)

Examples:
- 2 ints:    2 × 4 = 8 bytes
- 3 floats:  3 × 4 = 12 bytes
- 5 doubles: 5 × 8 = 40 bytes
- 100 chars: 100 × 1 = 100 bytes
```

---

## Key Takeaways

1. **Use arrays** for multiple related values
2. **Allocate N × sizeof(type)** bytes for N elements
3. **Array indexing** (`data[i]`) to access elements
4. **Single cudaMemcpy** can transfer entire array
5. **Space in format string** for space-separated output

---

## Practice Exercises

1. Read and echo **three** integers
2. Read two integers and print them in **reverse order**
3. Read N integers (N given first) and echo them all
4. Read two integers and print their **sum** alongside original values

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/20)*
