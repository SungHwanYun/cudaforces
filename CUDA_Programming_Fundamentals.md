# CUDA Programming Fundamentals

## A Comprehensive Guide for CUDA Online Judge

---

# Chapter 1: Introduction to Heterogeneous Parallel Computing

## 1.1 What is Heterogeneous Computing?

Modern computing systems are **heterogeneous** — they contain multiple types of processors working together. The most common configuration pairs a CPU (Central Processing Unit) with a GPU (Graphics Processing Unit).

```
┌─────────────────────────────────────────────────────────────┐
│                 Heterogeneous Computing System              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐              ┌─────────────────────┐      │
│   │     CPU     │              │        GPU          │      │
│   │   (Host)    │◄────────────►│     (Device)        │      │
│   │             │   PCIe Bus   │                     │      │
│   │  4-16 cores │              │  1000s of cores     │      │
│   │  Complex    │              │  Simple but many    │      │
│   │  Serial     │              │  Parallel           │      │
│   └─────────────┘              └─────────────────────┘      │
│         │                              │                    │
│         ▼                              ▼                    │
│   ┌─────────────┐              ┌─────────────────────┐      │
│   │ System RAM  │              │    GPU Memory       │      │
│   │  (Host)     │              │    (Device)         │      │
│   │  8-64 GB    │              │    4-48 GB          │      │
│   └─────────────┘              └─────────────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### CPU vs GPU: Design Philosophy

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Core Count** | 4-64 cores | 1000-10000+ cores |
| **Core Complexity** | Complex, out-of-order | Simple, in-order |
| **Clock Speed** | 3-5 GHz | 1-2 GHz |
| **Cache per Core** | Large (MB) | Small (KB) |
| **Optimized for** | Latency | Throughput |
| **Best at** | Serial tasks | Parallel tasks |

### Why GPUs for Computing?

GPUs were originally designed for rendering graphics, where the same operation (like calculating pixel color) must be applied to millions of pixels independently. This **Single Instruction, Multiple Data (SIMD)** pattern is common in many computational domains:

- **Scientific Computing**: Matrix operations, simulations
- **Machine Learning**: Neural network training and inference
- **Signal Processing**: FFT, convolution, filtering
- **Computer Vision**: Image processing, feature detection
- **Finance**: Monte Carlo simulations, risk analysis

## 1.2 The CUDA Platform

**CUDA** (Compute Unified Device Architecture) is NVIDIA's platform for general-purpose GPU computing. It provides:

1. **Programming Model**: Extensions to C/C++ for GPU programming
2. **Compiler**: nvcc compiles CUDA code
3. **Runtime API**: Functions for memory management and kernel launching
4. **Libraries**: cuBLAS, cuDNN, cuFFT, etc.

### CUDA Terminology

| Term | Definition |
|------|------------|
| **Host** | The CPU and its memory |
| **Device** | The GPU and its memory |
| **Kernel** | A function that runs on the GPU |
| **Thread** | Single execution unit on GPU |
| **Block** | Group of threads that can cooperate |
| **Grid** | Collection of all blocks for a kernel |

## 1.3 A Simple CUDA Program

Let's examine the structure of a basic CUDA program:

```cuda
// Kernel definition - runs on GPU
__global__ void addKernel(int* a, int* b, int* c) {
    int idx = threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

int main() {
    // 1. Declare host and device pointers
    int h_a[3] = {1, 2, 3};
    int h_b[3] = {4, 5, 6};
    int h_c[3];
    int *d_a, *d_b, *d_c;
    
    // 2. Allocate device memory
    cudaMalloc(&d_a, 3 * sizeof(int));
    cudaMalloc(&d_b, 3 * sizeof(int));
    cudaMalloc(&d_c, 3 * sizeof(int));
    
    // 3. Copy data from host to device
    cudaMemcpy(d_a, h_a, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 3 * sizeof(int), cudaMemcpyHostToDevice);
    
    // 4. Launch kernel
    addKernel<<<1, 3>>>(d_a, d_b, d_c);
    
    // 5. Copy result from device to host
    cudaMemcpy(h_c, d_c, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 6. Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}
```

### Program Flow

```
┌────────────────────────────────────────────────────────────┐
│                    CUDA Program Flow                       │
└────────────────────────────────────────────────────────────┘

    HOST (CPU)                         DEVICE (GPU)
    ──────────                         ────────────
        │
        ▼
   ┌─────────────┐
   │ Allocate    │
   │ host memory │
   └─────────────┘
        │
        ▼
   ┌─────────────┐                    ┌─────────────┐
   │ cudaMalloc  │ ──────────────────►│ Allocate    │
   │             │                    │ device mem  │
   └─────────────┘                    └─────────────┘
        │
        ▼
   ┌─────────────┐                    ┌─────────────┐
   │ cudaMemcpy  │ ═══════════════════►│ Data copied │
   │ H → D       │    PCIe Transfer   │ to device   │
   └─────────────┘                    └─────────────┘
        │
        ▼
   ┌─────────────┐                    ┌─────────────┐
   │ Launch      │ ──────────────────►│ Kernel      │
   │ kernel<<<>>>│                    │ executes    │
   └─────────────┘                    │ in parallel │
        │                             └─────────────┘
        │ (CPU continues                    │
        │  or waits)                        │
        ▼                                   ▼
   ┌─────────────┐                    ┌─────────────┐
   │ cudaMemcpy  │◄═══════════════════│ Results     │
   │ D → H       │    PCIe Transfer   │ ready       │
   └─────────────┘                    └─────────────┘
        │
        ▼
   ┌─────────────┐                    ┌─────────────┐
   │ cudaFree    │ ──────────────────►│ Free device │
   │             │                    │ memory      │
   └─────────────┘                    └─────────────┘
        │
        ▼
   ┌─────────────┐
   │ Use results │
   └─────────────┘
```

---

# Chapter 2: CUDA Programming Model

## 2.1 Kernels and Thread Hierarchy

### The __global__ Qualifier

Functions that run on the GPU are marked with `__global__`:

```cuda
__global__ void myKernel(int* data) {
    // This code runs on the GPU
    int idx = threadIdx.x;
    data[idx] = data[idx] * 2;
}
```

### Function Qualifiers

| Qualifier | Executes on | Called from |
|-----------|-------------|-------------|
| `__global__` | Device (GPU) | Host (CPU) |
| `__device__` | Device | Device |
| `__host__` | Host | Host |

```cuda
__device__ int square(int x) {
    return x * x;  // Helper function on GPU
}

__global__ void kernel(int* data) {
    data[threadIdx.x] = square(data[threadIdx.x]);
}
```

### Thread Hierarchy: Grids, Blocks, and Threads

CUDA organizes parallel execution in a three-level hierarchy:

```
┌─────────────────────────────────────────────────────────────┐
│                          GRID                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   BLOCK     │  │   BLOCK     │  │   BLOCK     │          │
│  │   (0,0)     │  │   (1,0)     │  │   (2,0)     │          │
│  │ ┌─┬─┬─┬─┐   │  │ ┌─┬─┬─┬─┐   │  │ ┌─┬─┬─┬─┐   │          │
│  │ │T│T│T│T│   │  │ │T│T│T│T│   │  │ │T│T│T│T│   │          │
│  │ ├─┼─┼─┼─┤   │  │ ├─┼─┼─┼─┤   │  │ ├─┼─┼─┼─┤   │          │
│  │ │T│T│T│T│   │  │ │T│T│T│T│   │  │ │T│T│T│T│   │          │
│  │ └─┴─┴─┴─┘   │  │ └─┴─┴─┴─┘   │  │ └─┴─┴─┴─┘   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   BLOCK     │  │   BLOCK     │  │   BLOCK     │          │
│  │   (0,1)     │  │   (1,1)     │  │   (2,1)     │          │
│  │ ┌─┬─┬─┬─┐   │  │ ┌─┬─┬─┬─┐   │  │ ┌─┬─┬─┬─┐   │          │
│  │ │T│T│T│T│   │  │ │T│T│T│T│   │  │ │T│T│T│T│   │          │
│  │ ├─┼─┼─┼─┤   │  │ ├─┼─┼─┼─┤   │  │ ├─┼─┼─┼─┤   │          │
│  │ │T│T│T│T│   │  │ │T│T│T│T│   │  │ │T│T│T│T│   │          │
│  │ └─┴─┴─┴─┘   │  │ └─┴─┴─┴─┘   │  │ └─┴─┴─┴─┘   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
│  Grid Dimensions: 3 × 2 blocks                              │
│  Block Dimensions: 4 × 2 threads                            │
│  Total Threads: 3 × 2 × 4 × 2 = 48 threads                  │
└─────────────────────────────────────────────────────────────┘
```

### Built-in Variables

CUDA provides built-in variables to identify each thread:

| Variable | Type | Description |
|----------|------|-------------|
| `threadIdx` | dim3 | Thread index within block |
| `blockIdx` | dim3 | Block index within grid |
| `blockDim` | dim3 | Dimensions of each block |
| `gridDim` | dim3 | Dimensions of the grid |

```cuda
__global__ void kernel() {
    // 1D indexing
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2D indexing
    int row = threadIdx.y;
    int col = threadIdx.x;
    int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
    int globalCol = blockIdx.x * blockDim.x + threadIdx.x;
}
```

## 2.2 Kernel Launch Configuration

### The <<<>>> Syntax

Kernels are launched with execution configuration:

```cuda
kernel<<<gridDim, blockDim>>>(arguments);
kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(arguments);
```

### 1D Launch

```cuda
// Launch 256 threads in a single block
kernel<<<1, 256>>>(data);

// Launch 1024 threads across 4 blocks
kernel<<<4, 256>>>(data);

// Process n elements
int threadsPerBlock = 256;
int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<numBlocks, threadsPerBlock>>>(data, n);
```

### 2D Launch with dim3

```cuda
// 2D block of threads
dim3 threadsPerBlock(16, 16);  // 256 threads total

// 2D grid of blocks
dim3 numBlocks(
    (width + 15) / 16,
    (height + 15) / 16
);

kernel<<<numBlocks, threadsPerBlock>>>(data, width, height);
```

### Computing Global Index

```cuda
// 1D global index
__global__ void kernel1D(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2;
    }
}

// 2D global index
__global__ void kernel2D(int* data, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;  // Row-major layout
        data[idx] = data[idx] * 2;
    }
}
```

## 2.3 Memory Management

### cudaMalloc and cudaFree

```cuda
int* d_data;

// Allocate device memory
cudaMalloc(&d_data, n * sizeof(int));

// Use the memory...

// Free device memory
cudaFree(d_data);
```

### cudaMemcpy

```cuda
// Host to Device
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// Device to Host
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

// Device to Device
cudaMemcpy(d_dest, d_src, size, cudaMemcpyDeviceToDevice);
```

### Memory Transfer Direction

```
┌────────────────────────────────────────────────────────────┐
│                   Memory Transfers                         │
└────────────────────────────────────────────────────────────┘

      HOST                              DEVICE
   ┌─────────┐                       ┌─────────┐
   │  h_src  │ ════════════════════► │  d_dst  │
   │         │  cudaMemcpyHostToDevice│         │
   └─────────┘                       └─────────┘

   ┌─────────┐                       ┌─────────┐
   │  h_dst  │ ◄════════════════════ │  d_src  │
   │         │  cudaMemcpyDeviceToHost│         │
   └─────────┘                       └─────────┘

                                     ┌─────────┐
                                     │  d_src  │
                                     │    ║    │
                                     │    ▼    │
                                     │  d_dst  │
                                     └─────────┘
                                cudaMemcpyDeviceToDevice
```

## 2.4 Error Handling

Always check for CUDA errors in production code:

```cuda
cudaError_t err;

// Check malloc
err = cudaMalloc(&d_data, size);
if (err != cudaSuccess) {
    printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
    return -1;
}

// Check kernel launch
kernel<<<blocks, threads>>>(data);
err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
}

// Check memcpy
err = cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
if (err != cudaSuccess) {
    printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
}
```

---

# Chapter 3: CUDA Execution Model

## 3.1 Hardware Architecture

### Streaming Multiprocessors (SMs)

A GPU contains multiple **Streaming Multiprocessors (SMs)**. Each SM can execute multiple thread blocks concurrently:

```
┌─────────────────────────────────────────────────────────────┐
│                         GPU                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │       SM 0       │  │       SM 1       │                 │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │                 │
│  │ │ CUDA Cores   │ │  │ │ CUDA Cores   │ │                 │
│  │ │ (64-128)     │ │  │ │ (64-128)     │ │                 │
│  │ └──────────────┘ │  │ └──────────────┘ │                 │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │                 │
│  │ │ Shared Mem   │ │  │ │ Shared Mem   │ │                 │
│  │ │ (48-164 KB)  │ │  │ │ (48-164 KB)  │ │                 │
│  │ └──────────────┘ │  │ └──────────────┘ │                 │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │                 │
│  │ │ L1 Cache     │ │  │ │ L1 Cache     │ │                 │
│  │ └──────────────┘ │  │ └──────────────┘ │                 │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │                 │
│  │ │ Registers    │ │  │ │ Registers    │ │                 │
│  │ │ (64K 32-bit) │ │  │ │ (64K 32-bit) │ │                 │
│  │ └──────────────┘ │  │ └──────────────┘ │                 │
│  └──────────────────┘  └──────────────────┘                 │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │       SM 2       │  │       SM n       │                 │
│  │       ...        │  │       ...        │                 │
│  └──────────────────┘  └──────────────────┘                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    L2 Cache                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Global Memory (GDDR/HBM)               │    │
│  │                   (4-80 GB)                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 3.2 Warps: The Unit of Execution

### What is a Warp?

A **warp** is a group of 32 threads that execute together in lockstep. All threads in a warp execute the same instruction at the same time (SIMT - Single Instruction, Multiple Threads).

```
┌─────────────────────────────────────────────────────────────┐
│                    Thread Block (128 threads)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Warp 0: Threads 0-31                                │    │
│  │ ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐    │
│  │ │0│1│2│3│4│5│6│7│8│9│...                         │31│    │
│  │ └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Warp 1: Threads 32-63                               │    │
│  │ ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐   │    │
│  │ │32│33│34│35│36│37│...                       │63│   │    │
│  │ └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Warp 2: Threads 64-95                               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Warp 3: Threads 96-127                              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Warp Divergence

When threads in a warp take different paths through conditional code, **warp divergence** occurs:

```cuda
__global__ void kernel(int* data) {
    int idx = threadIdx.x;
    
    if (idx % 2 == 0) {
        data[idx] = 1;    // Even threads execute this
    } else {
        data[idx] = 2;    // Odd threads execute this
    }
    // Both paths are serialized!
}
```

```
Warp Execution with Divergence:
───────────────────────────────
Time ──►

     ┌────────────────────┐ ┌────────────────────┐
     │   if-branch        │ │   else-branch      │
     │   (even threads)   │ │   (odd threads)    │
     │   active           │ │   active           │
     └────────────────────┘ └────────────────────┘
     
     Threads 0,2,4,6...      Threads 1,3,5,7...
     execute                 wait (masked)
     
                             Threads 1,3,5,7...
     Threads 0,2,4,6...      execute
     wait (masked)
```

**Best Practice**: Minimize divergence by grouping similar work together.

## 3.3 Block Scheduling

### How Blocks are Scheduled

The GPU scheduler assigns thread blocks to SMs as resources become available:

```
┌─────────────────────────────────────────────────────────────┐
│                    Block Scheduling                         │
└─────────────────────────────────────────────────────────────┘

Grid with 8 blocks:
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │
└────┴────┴────┴────┴────┴────┴────┴────┘

GPU with 2 SMs (each can run 2 blocks):

Time 0:
SM0: [B0] [B1]    SM1: [B2] [B3]

Time 1 (B0, B2 finish):
SM0: [B4] [B1]    SM1: [B5] [B3]

Time 2 (B1, B3 finish):
SM0: [B4] [B6]    SM1: [B5] [B7]

...and so on
```

### Occupancy

**Occupancy** is the ratio of active warps to maximum possible warps on an SM. Higher occupancy can hide memory latency:

```
Occupancy = Active Warps / Maximum Warps

Factors limiting occupancy:
- Registers per thread
- Shared memory per block
- Threads per block
```

## 3.4 Synchronization

### __syncthreads()

Synchronizes all threads within a block:

```cuda
__global__ void kernel() {
    __shared__ int data[256];
    
    // Phase 1: All threads write
    data[threadIdx.x] = threadIdx.x;
    
    __syncthreads();  // Wait for all writes to complete
    
    // Phase 2: All threads read (now safe)
    int value = data[255 - threadIdx.x];
}
```

**Warning**: All threads in the block must reach `__syncthreads()`. Conditional usage can cause deadlock:

```cuda
// DANGEROUS - can cause deadlock!
if (threadIdx.x < 16) {
    __syncthreads();  // Only some threads call this
}

// CORRECT
__syncthreads();  // All threads call this
if (threadIdx.x < 16) {
    // Do conditional work
}
```

### Atomic Operations

For thread-safe updates to shared or global memory:

```cuda
__global__ void histogram(int* data, int* bins, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&bins[data[idx]], 1);
    }
}
```

Common atomic operations:
- `atomicAdd()`, `atomicSub()`
- `atomicMax()`, `atomicMin()`
- `atomicAnd()`, `atomicOr()`, `atomicXor()`
- `atomicExch()`, `atomicCAS()`

---

# Chapter 4: CUDA Memory Hierarchy

## 4.1 Memory Types Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   CUDA Memory Hierarchy                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    Per-Thread                       │    │
│  │  ┌─────────────┐  ┌─────────────────────────────┐   │    │
│  │  │ Registers   │  │ Local Memory                │   │    │
│  │  │ (Fastest)   │  │ (Slow - in global)          │   │    │
│  │  │ ~1 cycle    │  │ For spilled registers       │   │    │
│  │  └─────────────┘  └─────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    Per-Block                        │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │              Shared Memory                  │    │    │
│  │  │              ~5 cycles                      │    │    │
│  │  │              48-164 KB per SM               │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    Per-Device                       │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐   │    │
│  │  │ Global Mem  │  │ Constant    │  │ Texture    │   │    │
│  │  │ ~400 cycles │  │ Memory      │  │ Memory     │   │    │
│  │  │ 4-80 GB     │  │ 64 KB       │  │ (Cached)   │   │    │
│  │  │ R/W         │  │ Read-only   │  │ Read-only  │   │    │
│  │  └─────────────┘  └─────────────┘  └────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Memory Comparison Table

| Memory Type | Location | Scope | Lifetime | Speed | Size |
|-------------|----------|-------|----------|-------|------|
| **Register** | On-chip | Thread | Thread | ~1 cycle | ~256 KB/SM |
| **Shared** | On-chip | Block | Block | ~5 cycles | 48-164 KB/SM |
| **Global** | Off-chip | Grid | Application | ~400 cycles | 4-80 GB |
| **Constant** | Off-chip (cached) | Grid | Application | ~5 cycles* | 64 KB |
| **Local** | Off-chip | Thread | Thread | ~400 cycles | - |

*When cached; ~400 cycles on cache miss

## 4.2 Global Memory

### Characteristics

- **Largest** memory space on GPU
- **Slowest** access (~400-800 cycles)
- Accessible by all threads in all blocks
- Persists for lifetime of application
- Allocated with `cudaMalloc()`, freed with `cudaFree()`

### Coalesced Memory Access

For best performance, adjacent threads should access adjacent memory locations:

```cuda
// GOOD: Coalesced access
// Threads 0,1,2,3 access data[0,1,2,3]
__global__ void coalesced(int* data) {
    int idx = threadIdx.x;
    data[idx] = data[idx] + 1;
}

// BAD: Strided access
// Threads 0,1,2,3 access data[0,32,64,96]
__global__ void strided(int* data) {
    int idx = threadIdx.x * 32;  // Stride of 32
    data[idx] = data[idx] + 1;
}
```

```
Coalesced Access (GOOD):
Thread:   0    1    2    3    4    5    6    7
          │    │    │    │    │    │    │    │
Memory:  ┌─┬──┬──┬──┬──┬──┬──┬──┐
         │0│ 1│ 2│ 3│ 4│ 5│ 6│ 7│
         └─┴──┴──┴──┴──┴──┴──┴──┘
          ▲    ▲    ▲    ▲    ▲    ▲    ▲    ▲
          └────┴────┴────┴────┴────┴────┴────┘
                   ONE memory transaction

Strided Access (BAD):
Thread:   0              1              2
          │              │              │
Memory:  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
         │ 0│  │  │  │  │32│  │  │  │  │64│  │  │
         └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
          ▲              ▲              ▲
          └──────────────┴──────────────┘
              MULTIPLE memory transactions
```

### Row-Major Layout for 2D Data

```cuda
// 2D array stored in 1D memory (row-major)
// Element at (row, col) = array[row * width + col]

__global__ void process2D(int* data, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;  // Row-major index
        data[idx] = data[idx] * 2;
    }
}
```

## 4.3 Shared Memory

### Characteristics

- **Fast** on-chip memory (~5 cycles)
- Shared by all threads in a block
- Limited size (48-164 KB per SM)
- Lifetime of thread block
- Must be explicitly managed

### Static Allocation

```cuda
__global__ void kernel() {
    __shared__ int sharedData[256];  // Static: size known at compile time
    
    sharedData[threadIdx.x] = threadIdx.x;
    __syncthreads();
    
    // Use shared data...
}
```

### Dynamic Allocation

```cuda
extern __shared__ int dynamicShared[];  // Declare without size

__global__ void kernel(int* data, int n) {
    // dynamicShared now points to shared memory
    dynamicShared[threadIdx.x] = data[threadIdx.x];
    __syncthreads();
    // ...
}

// Launch with shared memory size
kernel<<<blocks, threads, n * sizeof(int)>>>(data, n);
```

### Example: Parallel Reduction with Shared Memory

```cuda
__global__ void reduce(int* input, int* output, int n) {
    __shared__ int sdata[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load from global to shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

### Bank Conflicts

Shared memory is divided into 32 **banks**. Simultaneous access to different addresses in the same bank causes **bank conflicts**:

```
Shared Memory Banks (32 banks):
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ B0 │ B1 │ B2 │ B3 │ B4 │...│B30 │B31 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│  0 │  1 │  2 │  3 │  4 │...│ 30 │ 31 │  ◄─ Address 0-31
├────┼────┼────┼────┼────┼────┼────┼────┤
│ 32 │ 33 │ 34 │ 35 │ 36 │...│ 62 │ 63 │  ◄─ Address 32-63
├────┼────┼────┼────┼────┼────┼────┼────┤
│ 64 │ 65 │ 66 │ 67 │ 68 │...│ 94 │ 95 │  ◄─ Address 64-95
└────┴────┴────┴────┴────┴────┴────┴────┘

No conflict:           2-way conflict:
Thread 0 → Bank 0      Thread 0 → Bank 0 (addr 0)
Thread 1 → Bank 1      Thread 1 → Bank 0 (addr 32)  ← Same bank!
Thread 2 → Bank 2      Thread 2 → Bank 1
...                    ...
```

```cuda
// Potential bank conflict with stride of 32
__shared__ float data[32][32];
float val = data[threadIdx.x][0];  // All threads access same bank!

// Solution: Add padding
__shared__ float data[32][33];  // 33 instead of 32
float val = data[threadIdx.x][0];  // Now different banks
```

## 4.4 Constant Memory

### Characteristics

- **64 KB** total
- **Read-only** from device code
- **Cached** - very fast when all threads read same address
- Broadcast capability - one read serves all threads
- Best for data that doesn't change during kernel execution

### Usage

```cuda
// Declare in global scope
__constant__ float constData[256];

// Initialize from host
float hostData[256] = {...};
cudaMemcpyToSymbol(constData, hostData, sizeof(hostData));

// Use in kernel
__global__ void kernel(float* output) {
    int idx = threadIdx.x;
    output[idx] = constData[idx] * 2.0f;  // Read from constant memory
}
```

### Best Use Cases

- Lookup tables
- Kernel parameters
- Coefficients for convolution
- Configuration values

```cuda
// Example: Convolution kernel in constant memory
__constant__ float kernel[9];  // 3x3 convolution kernel

__global__ void convolve(float* input, float* output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col > 0 && col < width-1 && row > 0 && row < height-1) {
        float sum = 0.0f;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                sum += input[(row+i)*width + (col+j)] * kernel[(i+1)*3 + (j+1)];
            }
        }
        output[row*width + col] = sum;
    }
}
```

---

# Chapter 5: Putting It All Together

## 5.1 Choosing the Right Memory

```
┌─────────────────────────────────────────────────────────────┐
│                 Memory Selection Guide                      │
└─────────────────────────────────────────────────────────────┘

                    ┌───────────────────┐
                    │   What data?      │
                    └─────────┬─────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
    ┌───────────┐      ┌───────────┐      ┌───────────┐
    │ Per-thread│      │  Shared   │      │  Global   │
    │   data    │      │   data    │      │   data    │
    └─────┬─────┘      └─────┬─────┘      └─────┬─────┘
          │                  │                  │
          ▼                  ▼                  ▼
    ┌───────────┐      ┌───────────┐      ┌───────────┐
    │ Registers │      │ Shared    │      │ Global    │
    │           │      │ Memory    │      │ Memory    │
    └───────────┘      └───────────┘      └───────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Read-only across  │
                    │ all threads?      │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │ Yes               │ No
                    ▼                   ▼
              ┌───────────┐      ┌───────────┐
              │ Constant  │      │ Keep in   │
              │ Memory    │      │ Shared    │
              └───────────┘      └───────────┘
```

## 5.2 Common Patterns

### Pattern 1: Element-wise Operations

Perfect for GPU - no communication needed between threads.

```cuda
// Vector operations, ReLU, scaling, etc.
__global__ void elementwise(float* A, float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}
```

### Pattern 2: Reduction

Combine all elements into single value using shared memory.

```cuda
__global__ void reduce(float* input, float* output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

### Pattern 3: Stencil/Convolution

Use shared memory to cache neighboring elements.

```cuda
__global__ void stencil1D(float* in, float* out, int n) {
    __shared__ float s[BLOCK_SIZE + 2];  // Include halo
    
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int lidx = threadIdx.x + 1;  // Leave room for halo
    
    // Load main data
    s[lidx] = in[gidx];
    
    // Load halo elements
    if (threadIdx.x == 0 && gidx > 0)
        s[0] = in[gidx - 1];
    if (threadIdx.x == blockDim.x - 1 && gidx < n - 1)
        s[lidx + 1] = in[gidx + 1];
    
    __syncthreads();
    
    // Compute stencil
    if (gidx > 0 && gidx < n - 1)
        out[gidx] = 0.25f * s[lidx-1] + 0.5f * s[lidx] + 0.25f * s[lidx+1];
}
```

### Pattern 4: Matrix Operations

Use 2D indexing and tiled algorithms for large matrices.

```cuda
__global__ void matrixAdd(float* A, float* B, float* C, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

// Launch
dim3 threads(16, 16);
dim3 blocks((width+15)/16, (height+15)/16);
matrixAdd<<<blocks, threads>>>(A, B, C, width, height);
```

## 5.3 Performance Guidelines

### Do's

1. **Maximize parallelism**: Use many threads
2. **Coalesce memory access**: Adjacent threads → adjacent memory
3. **Use shared memory**: For data reuse within a block
4. **Minimize divergence**: Group similar work together
5. **Overlap computation and memory**: Use streams

### Don'ts

1. **Don't transfer unnecessarily**: Minimize host↔device transfers
2. **Don't use small blocks**: At least 128-256 threads per block
3. **Don't ignore occupancy**: More active warps hide latency
4. **Don't serialize with atomics**: Use reduction patterns instead
5. **Don't forget bounds checking**: Always verify index < size

## 5.4 Quick Reference

### Memory Operations

```cuda
// Allocation
cudaMalloc(&d_ptr, size);
cudaFree(d_ptr);

// Transfer
cudaMemcpy(dst, src, size, direction);
// direction: cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost

// Constant memory
cudaMemcpyToSymbol(constVar, hostPtr, size);
```

### Kernel Launch

```cuda
kernel<<<numBlocks, threadsPerBlock>>>(args);
kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(args);

// 2D
dim3 blocks(bx, by);
dim3 threads(tx, ty);
kernel<<<blocks, threads>>>(args);
```

### Thread Indexing

```cuda
// 1D
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int idx = row * width + col;  // Row-major

// 2D block-local
int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
```

### Synchronization

```cuda
__syncthreads();  // Block-level sync
atomicAdd(&var, val);  // Atomic operations
```

---

## Summary

This guide covered the essential concepts for CUDA programming:

1. **Heterogeneous Computing**: CPU+GPU working together
2. **Programming Model**: Kernels, threads, blocks, grids
3. **Execution Model**: Warps, SMs, scheduling
4. **Memory Hierarchy**: Global, shared, constant, registers
5. **Common Patterns**: Element-wise, reduction, stencil, matrix

With these fundamentals, you're ready to solve problems on the CUDA Online Judge!

---

*This document is part of the [CUDA Online Judge](https://cudaforces.com) educational materials.*
