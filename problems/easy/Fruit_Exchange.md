# Fruit Exchange

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

You have n apples and want to exchange them for tangerines through a series of trades.

The exchange rates are:
- 3 apples → a pears
- 5 pears → b persimmons
- 3 persimmons → c tangerines

Given n, a, b, and c, find the maximum number of tangerines you can obtain starting with n apples.

Note: You can only perform each exchange when you have enough fruits. Leftover fruits from incomplete exchanges cannot be used.

### Input
A single line containing four integers n, a, b, and c, separated by spaces.

**Constraints:**
- 1 ≤ n ≤ 100
- 1 ≤ a, b, c ≤ 10

### Output
Print the maximum number of tangerines obtainable.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 9 2 3 4 | 4 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 30 5 2 3 | 18 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 2 10 10 10 | 0 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void fruitExchangeKernel(int* data, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int n = data[0];  // apples
        int a = data[1];  // pears per 3 apples
        int b = data[2];  // persimmons per 5 pears
        int c = data[3];  // tangerines per 3 persimmons
        
        // Step 1: Apples → Pears
        int pears = (n / 3) * a;
        
        // Step 2: Pears → Persimmons
        int persimmons = (pears / 5) * b;
        
        // Step 3: Persimmons → Tangerines
        int tangerines = (persimmons / 3) * c;
        
        *result = tangerines;
    }
}

int main() {
    int n, a, b, c;
    scanf("%d %d %d %d", &n, &a, &b, &c);
    
    int h_data[4] = {n, a, b, c};
    int h_result;
    
    // Device memory
    int *d_data, *d_result;
    cudaMalloc(&d_data, 4 * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    // Copy input to device
    cudaMemcpy(d_data, h_data, 4 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    fruitExchangeKernel<<<1, 1>>>(d_data, d_result);
    
    // Copy result back to host
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d\n", h_result);
    
    cudaFree(d_data);
    cudaFree(d_result);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void fruitExchangeKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs chained exchange calculation on GPU |

---

## CUDA Concepts Covered

### 1. Chained Exchange Logic

Each exchange follows the pattern:
```cuda
output = (input / required) * received
```

```cuda
// Step 1: 3 apples → a pears
int pears = (n / 3) * a;

// Step 2: 5 pears → b persimmons
int persimmons = (pears / 5) * b;

// Step 3: 3 persimmons → c tangerines
int tangerines = (persimmons / 3) * c;
```

### 2. Integer Division for "Complete Exchanges"

The `/` operator in C automatically handles incomplete exchanges:

```cuda
// If you have 9 apples and need 3 per exchange:
9 / 3 = 3  // Can exchange 3 times

// If you have 2 apples and need 3 per exchange:
2 / 3 = 0  // Cannot exchange at all (leftover is lost)
```

### 3. Visualization

**Example 1: n=9, a=2, b=3, c=4**

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  9 Apples   │ ──→  │  6 Pears    │ ──→  │ 3 Persimmon │ ──→  │ 4 Tangerine │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
       │                    │                    │                    │
   9 ÷ 3 = 3           6 ÷ 5 = 1           3 ÷ 3 = 1              RESULT
   3 × 2 = 6           1 × 3 = 3           1 × 4 = 4
```

**Example 2: n=30, a=5, b=2, c=3**

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ 30 Apples   │ ──→  │ 50 Pears    │ ──→  │20 Persimmon │ ──→  │18 Tangerine │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
       │                    │                    │                    │
  30 ÷ 3 = 10         50 ÷ 5 = 10         20 ÷ 3 = 6              RESULT
  10 × 5 = 50         10 × 2 = 20          6 × 3 = 18
                                          (2 leftover)
```

**Example 3: n=2, a=10, b=10, c=10**

```
┌─────────────┐      ┌─────────────┐
│  2 Apples   │ ──→  │  0 Pears    │ ──→ ...  → 0 Tangerines
└─────────────┘      └─────────────┘
       │
   2 ÷ 3 = 0   ← Cannot even start!
   0 × 10 = 0
```

### 4. Exchange Formula

General formula for each step:
```
output_count = (input_count / required_input) × output_per_exchange
```

Combined formula:
```
tangerines = (((n / 3) * a) / 5 * b) / 3 * c
```

### 5. Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  n=9, a=2, b=3, c=4                                      │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (HostToDevice)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   Step 1: pears = (9/3) × 2 = 6                          │
│   Step 2: persimmons = (6/5) × 3 = 3                     │
│   Step 3: tangerines = (3/3) × 4 = 4                     │
│                                                          │
│   d_result: [4]                                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (DeviceToHost)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  result = 4 → printf("4\n")                              │
└──────────────────────────────────────────────────────────┘
```

---

## Alternative Solutions

### Using Single Expression

```cuda
__global__ void fruitExchangeKernel(int* data, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int n = data[0], a = data[1], b = data[2], c = data[3];
        
        // All steps in one expression
        *result = (((n / 3) * a) / 5 * b) / 3 * c;
    }
}
```

### Using printf in Kernel

```cuda
__global__ void fruitExchangeKernel(int n, int a, int b, int c) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int pears = (n / 3) * a;
        int persimmons = (pears / 5) * b;
        int tangerines = (persimmons / 3) * c;
        printf("%d\n", tangerines);
    }
}
```

### Step-by-Step with Debug Output

```cuda
__global__ void fruitExchangeKernel(int* data, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int n = data[0], a = data[1], b = data[2], c = data[3];
        
        int exchanges1 = n / 3;
        int pears = exchanges1 * a;
        
        int exchanges2 = pears / 5;
        int persimmons = exchanges2 * b;
        
        int exchanges3 = persimmons / 3;
        int tangerines = exchanges3 * c;
        
        *result = tangerines;
    }
}
```

---

## Exchange Summary Table

| Step | Input | Required | Output | Formula |
|------|-------|----------|--------|---------|
| 1 | Apples | 3 | a Pears | `(apples/3) × a` |
| 2 | Pears | 5 | b Persimmons | `(pears/5) × b` |
| 3 | Persimmons | 3 | c Tangerines | `(persimmons/3) × c` |

---

## Common Mistakes

### ❌ Using Float Division
```cuda
float pears = (n / 3.0) * a;  // Wrong! Keeps fractional exchanges
int pears = (n / 3) * a;      // Correct - integer division
```

### ❌ Wrong Order of Operations
```cuda
int pears = n / 3 * a;        // Correct (left to right)
int pears = n / (3 * a);      // Wrong! Different result
```

### ❌ Forgetting Intermediate Steps
```cuda
// Wrong - skipping persimmons
int tangerines = (pears / 3) * c;  // Should divide pears by 5, not 3

// Correct chain
int persimmons = (pears / 5) * b;
int tangerines = (persimmons / 3) * c;
```

### ❌ Confusing Exchange Rates
```cuda
// Exchange rate order: a, b, c
// a = pears per 3 apples
// b = persimmons per 5 pears
// c = tangerines per 3 persimmons
```

---

## Edge Cases

| n | a | b | c | Pears | Persimmons | Tangerines |
|---|---|---|---|-------|------------|------------|
| 2 | 10 | 10 | 10 | 0 | 0 | 0 |
| 3 | 1 | 1 | 1 | 1 | 0 | 0 |
| 15 | 5 | 1 | 1 | 25 | 5 | 1 |
| 100 | 10 | 10 | 10 | 330 | 660 | 2200 |

---

## Key Takeaways

1. **Chained exchanges** — output of one step becomes input of next
2. **Integer division** — automatically handles "leftover" fruits
3. **Order matters** — each step has different exchange rate
4. **Can result in 0** — if initial count is too low
5. **Multi-step problem** — break down into simple steps

---

## Practice Exercises

1. Add a **fourth exchange** type
2. Find the **minimum apples needed** to get at least 1 tangerine
3. Calculate **leftover fruits** at each step
4. Allow **reverse exchanges** (tangerines back to persimmons)

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/135)*
