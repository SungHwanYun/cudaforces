# Problem Number Finder

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

In contests hosted on **cudaforces**, problem IDs are represented using uppercase letters according to the following rule:

- Problem 1 is A, Problem 2 is B, Problem 3 is C, ..., Problem 26 is Z

Given a problem ID, output the corresponding problem number.

### Input
A single uppercase letter is given on the first line.

**Constraints:**
- The input is a single uppercase letter from A to Z.

### Output
Print the problem number corresponding to the given problem ID.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| A | 1 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| G | 7 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| Z | 26 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void convertKernel(char* ch, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*ch) - 'A' + 1;
    }
}

int main() {
    char ch;
    scanf(" %c", &ch);
    
    // Device memory
    char *d_ch;
    int *d_result;
    cudaMalloc(&d_ch, sizeof(char));
    cudaMalloc(&d_result, sizeof(int));
    
    // Copy input to device
    cudaMemcpy(d_ch, &ch, sizeof(char), cudaMemcpyHostToDevice);
    
    // Launch kernel
    convertKernel<<<1, 1>>>(d_ch, d_result);
    
    // Copy result back to host
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d\n", result);
    
    cudaFree(d_ch);
    cudaFree(d_result);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void convertKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs letter-to-number conversion on GPU |

---

## CUDA Concepts Covered

### 1. Character Arithmetic (Reverse)

Converting a letter to a 1-based index:

```cuda
*result = (*ch) - 'A' + 1;

// Examples:
// 'A' - 'A' + 1 = 0 + 1 = 1
// 'G' - 'A' + 1 = 6 + 1 = 7
// 'Z' - 'A' + 1 = 25 + 1 = 26
```

### 2. ASCII Subtraction

Subtracting characters gives their distance:

```
'A' = 65
'G' = 71

'G' - 'A' = 71 - 65 = 6
```

Then add 1 for 1-based indexing.

### 3. Why + 1?

Character subtraction gives 0-based result, but problem needs 1-based:

```
'A' - 'A' = 0  →  0 + 1 = 1
'B' - 'A' = 1  →  1 + 1 = 2
...
'Z' - 'A' = 25 →  25 + 1 = 26
```

### 4. Visualization

```
char:    'A'  'B'  'C'  'D'  'E'  'F'  'G'  ...  'Z'
          │    │    │    │    │    │    │         │
ch-'A':   0    1    2    3    4    5    6   ...   25
          │    │    │    │    │    │    │         │
+1:       1    2    3    4    5    6    7   ...   26
```

### 5. Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  ch = 'G'                                                │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (HostToDevice)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   d_ch: ['G']                                            │
│     │                                                    │
│     ▼                                                    │
│   'G' - 'A' + 1 = 6 + 1 = 7                              │
│     │                                                    │
│     ▼                                                    │
│   d_result: [7]                                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (DeviceToHost)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  result = 7 → printf("7\n")                              │
└──────────────────────────────────────────────────────────┘
```

---

## Alternative Solutions

### Direct printf in Kernel

```cuda
__global__ void convertKernel(char* ch) {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("%d\n", (*ch) - 'A' + 1);
    }
}
```

### Using ASCII Value Directly

```cuda
__global__ void convertKernel(char* ch, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // 'A' = 65
        *result = (*ch) - 64;  // 65 - 64 = 1
    }
}
```

### Step-by-Step Calculation

```cuda
__global__ void convertKernel(char* ch, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int offset = (*ch) - 'A';  // 0-based position
        *result = offset + 1;       // Convert to 1-based
    }
}
```

---

## Inverse Problem Comparison

| Problem | Direction | Formula | Example |
|---------|-----------|---------|---------|
| Problem ID Converter | Number → Letter | `'A' + (n - 1)` | 7 → 'G' |
| **Problem Number Finder** | Letter → Number | `ch - 'A' + 1` | 'G' → 7 |

These are **inverse operations**:
```
n → 'A' + (n - 1) = ch
ch - 'A' + 1 = n
```

---

## Letter-to-Number Mapping

| Letter | ch - 'A' | Result |
|--------|----------|--------|
| A | 0 | 1 |
| B | 1 | 2 |
| C | 2 | 3 |
| ... | ... | ... |
| G | 6 | 7 |
| ... | ... | ... |
| Z | 25 | 26 |

---

## Common Mistakes

### ❌ Forgetting to Add 1
```cuda
*result = (*ch) - 'A';     // Wrong! 'A' gives 0
*result = (*ch) - 'A' + 1; // Correct - 'A' gives 1
```

### ❌ Wrong Subtraction Order
```cuda
*result = 'A' - (*ch) + 1; // Wrong! Gives negative values
*result = (*ch) - 'A' + 1; // Correct
```

### ❌ Using Wrong Base Character
```cuda
*result = (*ch) - 'a' + 1; // Wrong! Input is uppercase
*result = (*ch) - 'A' + 1; // Correct - uppercase base
```

### ❌ Missing scanf Space
```cuda
scanf("%c", &ch);   // May read leftover newline
scanf(" %c", &ch);  // Correct - skips whitespace
```

---

## Round-Trip Verification

```cuda
// Number to Letter
char letter = 'A' + (7 - 1);  // 'G'

// Letter to Number
int number = 'G' - 'A' + 1;   // 7

// They are inverses!
```

---

## Key Takeaways

1. **Inverse of ID Converter** — reverses the previous problem
2. **Character subtraction** — gives distance between characters
3. **0-based to 1-based** — add 1 to convert
4. **scanf whitespace** — use `" %c"` to skip leading whitespace
5. **ASCII math** — 'G' - 'A' = 71 - 65 = 6

---

## Practice Exercises

1. Handle **lowercase** input (convert to uppercase first)
2. Convert a **string of letters** to numbers
3. Handle **two-letter IDs** (AA = 27, AB = 28, ...)
4. Validate input and handle **invalid characters**

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/143)*
