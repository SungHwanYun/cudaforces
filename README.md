# 🚀 CUDAForces

**A GPU Programming Learning Platform - Online Judge for CUDA**

[![CUDA](https://img.shields.io/badge/CUDA-Educational-76B900?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

CUDAForces is an educational online judge platform designed for learning GPU programming. Write and test CUDA code without requiring actual GPU hardware through our innovative CPU Transpiler technology.

---

## ✨ Features

- 🎓 **GPU-Free Learning** - Practice CUDA programming without expensive GPU hardware
- 🔄 **CPU Transpiler** - Automatically converts CUDA code to CPU-executable C++ with OpenMP
- ✅ **Correctness Verification** - Focus on algorithm correctness, not performance tuning
- 📝 **Comprehensive Validation** - Ensures proper CUDA programming patterns
- 🎯 **Educational Focus** - Learn memory management and parallel thinking from scratch

---

## 🔧 How It Works

### CPU Transpiler Architecture

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   CUDA Code     │      │   Transpiler    │      │   C++ + OpenMP  │
│   (.cu file)    │ ───► │   - Parser      │ ───► │   (Executable)  │
│                 │      │   - Validator   │      │                 │
│  __global__     │      │   - Generator   │      │  #pragma omp    │
│  <<<>>>         │      │                 │      │  parallel for   │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

Your CUDA code is transformed into equivalent C++ code that simulates GPU behavior:

| CUDA Element | CPU Conversion |
|--------------|----------------|
| `__global__` function | Regular C++ function |
| `<<<blocks, threads>>>` | OpenMP nested loops |
| `threadIdx`, `blockIdx` | Function parameters |
| `cudaMalloc` / `cudaFree` | `malloc` / `free` |
| `cudaMemcpy` | `memcpy` |
| `__shared__` memory | Per-block arrays |
| `atomicAdd` | `__atomic_fetch_add` |
| `__syncthreads()` | `#pragma omp barrier` |

> ⚠️ **Note**: This platform is for **correctness verification only**. Performance benchmarking is not meaningful as CPU emulation differs significantly from actual GPU execution.

---

## 📋 Supported CUDA Features

### Memory & Execution

| Category | Features |
|----------|----------|
| **Keywords** | `__global__`, `__device__`, `__host__`, `__shared__`, `__constant__` |
| **Built-ins** | `threadIdx`, `blockIdx`, `blockDim`, `gridDim`, `warpSize` |
| **Memory Ops** | `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemset`, `cudaMemcpyAsync` |
| **Sync** | `__syncthreads()`, `__syncwarp()`, `cudaDeviceSynchronize()` |

### Atomic Operations

```cuda
atomicAdd()    atomicSub()    atomicExch()   atomicMin()    atomicMax()
atomicInc()    atomicDec()    atomicCAS()    atomicAnd()    atomicOr()    atomicXor()
```

### Warp Primitives

```cuda
__shfl_sync()        __shfl_up_sync()      __shfl_down_sync()    __shfl_xor_sync()
__ballot_sync()      __all_sync()          __any_sync()          __activemask()
```

### Data Types

- Primitives: `int`, `float`, `double`, `char`, `long long`, `size_t`, `bool`
- Compound: `dim3`, `struct`, `enum`, `typedef`
- Pointers & Arrays: Full support including multi-dimensional

---

## ✅ Code Validation Rules

All submissions must pass these validations:

| Rule | Description | Error |
|------|-------------|-------|
| **Kernel Required** | Must have at least one `__global__` function | E3001 |
| **Kernel Called** | Kernel must be invoked with `<<<>>>` syntax | E3001 |
| **Meaningful Work** | Kernel must perform actual computation | E3002 |
| **Use Parallelism** | Must use `threadIdx`, `blockIdx`, etc. | E3003 |
| **GPU Memory** | Must use `cudaMalloc`, `cudaMemcpy` | E3004 |
| **No Forbidden Functions** | Cannot use `qsort`, `strcpy`, etc. | E3005 |
| **No STL Containers** | Cannot use `vector`, `string`, etc. | E3006 |

### Example: Valid vs Invalid Code

```cuda
// ❌ Invalid - No parallelism used
__global__ void bad_kernel(int* arr) {
    arr[0] = 1;  // All threads write to same location
}

// ✅ Valid - Proper parallel pattern
__global__ void good_kernel(int* arr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    arr[i] = i;  // Each thread processes different element
}
```

---

## 🚫 Prohibited Items

### Forbidden Functions

These must be **implemented manually** for educational purposes:

```c
// Sorting/Searching (stdlib.h)
qsort(), bsearch()

// String operations (string.h) - ALL PROHIBITED
strcpy(), strlen(), strcmp(), strcat(), memcpy(), memset(), ...

// STL algorithms
sort(), find(), copy(), fill(), reverse(), accumulate(), ...
```

### Forbidden Types

Use **C-style arrays and pointers** instead:

```cpp
// ❌ Prohibited
std::vector<int> arr;
std::string str;
std::map<int, int> m;

// ✅ Use instead
int* arr = (int*)malloc(n * sizeof(int));
char str[100];
// Implement your own data structure
```

---

## 🚀 Quick Start

### Basic Template

```cuda
#include <stdio.h>

__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1000;
    int size = n * sizeof(int);
    
    // Host memory
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);
    
    // Initialize
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Copy to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    // Copy result back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Print first 10 results
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }
    
    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    
    return 0;
}
```

### Shared Memory Reduction

```cuda
#define BLOCK_SIZE 256

__global__ void reduce(int* input, int* output, int n) {
    __shared__ int sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}
```

---

## 📁 Project Structure

```
cudaforces/
├── frontend/               # React web application
│   ├── src/
│   │   ├── components/    # UI components
│   │   ├── pages/         # Page components
│   │   └── services/      # API services
│   └── package.json
│
├── backend/                # Flask API server
│   ├── app/
│   │   ├── models/        # Database models
│   │   ├── routes/        # API endpoints
│   │   └── services/      # Business logic
│   └── requirements.txt
│
├── judge/                  # C++ judge server
│   ├── src/
│   │   ├── judge.cpp      # Main judge logic
│   │   └── sandbox.cpp    # Execution sandbox
│   └── CMakeLists.txt
│
├── transpiler/             # CUDA to C++ transpiler
│   ├── src/
│   │   ├── parser/        # CUDA parser
│   │   ├── transpiler/    # Code generator
│   │   └── validator/     # Code validator
│   ├── include/
│   │   ├── ast.h          # AST definitions
│   │   ├── lexer.h        # Lexer
│   │   ├── parser.h       # Parser
│   │   └── error.h        # Error codes
│   └── docs/
│       ├── GUIDE.md       # Development guide
│       └── README.md      # Transpiler docs
│
└── docs/
    ├── CODING_GUIDE_EN.md # English coding guide
    └── CODING_GUIDE_KO.md # Korean coding guide
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | React, TypeScript, Tailwind CSS |
| **Backend** | Flask, SQLAlchemy, JWT |
| **Judge Server** | C++17, OpenMP |
| **Transpiler** | C++17, Custom Parser |
| **Database** | PostgreSQL |
| **Infrastructure** | Docker, AWS |

---

## 📊 Error Code Reference

| Code | Category | Description |
|------|----------|-------------|
| E1xxx | Syntax | CUDA syntax errors (missing semicolon, brackets, etc.) |
| E2xxx | Semantic | CUDA semantic errors (wrong memcpy direction, invalid access) |
| E3xxx | Validation | Policy violations (no kernel, no parallelism, forbidden functions) |
| E4xxx | Not Supported | Unsupported CUDA features (dynamic parallelism, unified memory) |
| E5xxx | Internal | System errors (parser bug, transpiler issue) |

---

## 📖 Documentation

- [English Coding Guide](docs/CODING_GUIDE_EN.md)
- [한국어 코딩 가이드](docs/CODING_GUIDE_KO.md)
- [Transpiler Development Guide](transpiler/docs/GUIDE.md)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or support, please contact: **ejpark29@gmail.com**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Happy CUDA Learning! 🚀</b>
</p>
