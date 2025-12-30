# Minutes to Coding Time

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

A developer spent some time coding during the day. The coding duration is given in total minutes.

Convert the given minutes to HH:MM format. Hours should be at least 2 digits (with leading zero if necessary), and minutes must always be exactly 2 digits.

### Input
A single line containing an integer A, the total coding time in minutes.

**Constraints:**
- 0 ≤ A ≤ 10⁶

### Output
Print the coding time in HH:MM format.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 150 | 02:30 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 0 | 00:00 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 10000 | 166:40 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void minutesToTimeKernel(int* totalMinutes, int* hours, int* mins) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *hours = (*totalMinutes) / 60;
        *mins = (*totalMinutes) % 60;
    }
}

int main() {
    int total;
    scanf("%d", &total);
    
    // Device memory
    int *d_total, *d_hours, *d_mins;
    cudaMalloc(&d_total, sizeof(int));
    cudaMalloc(&d_hours, sizeof(int));
    cudaMalloc(&d_mins, sizeof(int));
    
    // Copy input to device
    cudaMemcpy(d_total, &total, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    minutesToTimeKernel<<<1, 1>>>(d_total, d_hours, d_mins);
    
    // Copy results back to host
    int hours, mins;
    cudaMemcpy(&hours, d_hours, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&mins, d_mins, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%02d:%02d\n", hours, mins);
    
    cudaFree(d_total);
    cudaFree(d_hours);
    cudaFree(d_mins);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void minutesToTimeKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs time conversion on GPU |

---

## CUDA Concepts Covered

### 1. Reverse Time Conversion

Convert total minutes to hours and minutes:

```cuda
*hours = (*totalMinutes) / 60;  // Integer division
*mins = (*totalMinutes) % 60;   // Remainder

// Examples:
// 150 → 150/60 = 2 hours, 150%60 = 30 minutes → 02:30
// 0 → 0/60 = 0 hours, 0%60 = 0 minutes → 00:00
// 10000 → 10000/60 = 166 hours, 10000%60 = 40 minutes → 166:40
```

### 2. Printf Format Specifiers

The `%02d` format ensures minimum 2 digits with leading zeros:

```cuda
printf("%02d:%02d\n", hours, mins);

// %02d breakdown:
// %  - format specifier start
// 0  - pad with zeros (not spaces)
// 2  - minimum width of 2 characters
// d  - integer type

// Examples:
// hours = 2  → "02"
// hours = 166 → "166" (no truncation, just minimum)
// mins = 5 → "05"
// mins = 30 → "30"
```

### 3. Visualization

```
Total: 150 minutes
         ↓
┌──────────────────────────────────────────┐
│  Division: 150 / 60 = 2 (hours)          │
│  Modulo:   150 % 60 = 30 (minutes)       │
│                                          │
│  Format: %02d → "02"                     │
│          %02d → "30"                     │
└──────────────────────────────────────────┘
         ↓
Output: "02:30"
```

### 4. Large Values (Example 3)

```
Total: 10000 minutes
         ↓
┌──────────────────────────────────────────┐
│  Division: 10000 / 60 = 166 (hours)      │
│  Modulo:   10000 % 60 = 40 (minutes)     │
│                                          │
│  Format: %02d → "166" (≥2 digits, OK)    │
│          %02d → "40"                     │
└──────────────────────────────────────────┘
         ↓
Output: "166:40"
```

### 5. Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  Input: 150 minutes                                      │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (HostToDevice)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   d_total: [150]                                         │
│        │                                                 │
│   ┌────┴────┐                                            │
│   │         │                                            │
│   /60      %60                                           │
│   │         │                                            │
│   ▼         ▼                                            │
│  d_hours  d_mins                                         │
│   [2]      [30]                                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (DeviceToHost)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  printf("%02d:%02d", 2, 30) → "02:30"                    │
└──────────────────────────────────────────────────────────┘
```

---

## Alternative Solutions

### Using Array

```cuda
__global__ void minutesToTimeKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int total = data[0];
        data[1] = total / 60;  // hours
        data[2] = total % 60;  // minutes
    }
}

int main() {
    int total;
    scanf("%d", &total);
    
    int h_data[3] = {total, 0, 0};
    
    int* d_data;
    cudaMalloc(&d_data, 3 * sizeof(int));
    cudaMemcpy(d_data, h_data, 3 * sizeof(int), cudaMemcpyHostToDevice);
    
    minutesToTimeKernel<<<1, 1>>>(d_data);
    
    cudaMemcpy(h_data, d_data, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%02d:%02d\n", h_data[1], h_data[2]);
    
    cudaFree(d_data);
    
    return 0;
}
```

### Direct Output in Kernel

```cuda
__global__ void minutesToTimeKernel(int* totalMinutes) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int hours = (*totalMinutes) / 60;
        int mins = (*totalMinutes) % 60;
        printf("%02d:%02d\n", hours, mins);
    }
}
```

---

## Printf Format Specifiers Reference

| Specifier | Meaning | Example |
|-----------|---------|---------|
| `%d` | Integer, no padding | `5` → "5" |
| `%2d` | Min 2 chars, space-padded | `5` → " 5" |
| `%02d` | Min 2 chars, zero-padded | `5` → "05" |
| `%3d` | Min 3 chars, space-padded | `5` → "  5" |
| `%03d` | Min 3 chars, zero-padded | `5` → "005" |

### Leading Zero Examples

```cuda
printf("%02d", 0);    // "00"
printf("%02d", 5);    // "05"
printf("%02d", 42);   // "42"
printf("%02d", 166);  // "166" (no truncation)
```

---

## Relationship with Previous Problem

| Problem | Direction | Formula |
|---------|-----------|---------|
| Time to Minutes | HH:MM → total | `hours × 60 + minutes` |
| **Minutes to Time** | total → HH:MM | `total / 60 : total % 60` |

These are **inverse operations**:

```
Forward:  02:30 → 2 × 60 + 30 = 150
Reverse:  150 → 150/60 : 150%60 = 02:30
```

---

## Common Mistakes

### ❌ Wrong Format (No Leading Zeros)
```cuda
printf("%d:%d\n", hours, mins);    // Wrong! "2:30" instead of "02:30"
printf("%02d:%02d\n", hours, mins); // Correct - "02:30"
```

### ❌ Reversed Division and Modulo
```cuda
*hours = (*totalMinutes) % 60;  // Wrong! This gives minutes
*mins = (*totalMinutes) / 60;   // Wrong! This gives hours
```

### ❌ Using Float Division
```cuda
float hours = (*totalMinutes) / 60.0f;  // Wrong! We need integer division
int hours = (*totalMinutes) / 60;       // Correct
```

### ❌ Truncating Large Hours
```cuda
// %02d does NOT truncate, only adds leading zeros
// hours = 166 → "166" (not "66")
```

---

## Edge Cases

| Input | Hours | Minutes | Output |
|-------|-------|---------|--------|
| 0 | 0 | 0 | 00:00 |
| 1 | 0 | 1 | 00:01 |
| 59 | 0 | 59 | 00:59 |
| 60 | 1 | 0 | 01:00 |
| 61 | 1 | 1 | 01:01 |
| 1439 | 23 | 59 | 23:59 |
| 1440 | 24 | 0 | 24:00 |
| 10000 | 166 | 40 | 166:40 |
| 1000000 | 16666 | 40 | 16666:40 |

---

## Key Takeaways

1. **Division gives hours** — `total / 60`
2. **Modulo gives minutes** — `total % 60`
3. **`%02d` format** — ensures at least 2 digits with leading zeros
4. **No truncation** — `%02d` is minimum width, not maximum
5. **Inverse of time-to-minutes** — reverse conversion pattern

---

## Practice Exercises

1. Handle **seconds** — convert total seconds to HH:MM:SS
2. Add **days** — output DD:HH:MM for very large values
3. Validate output — ensure minutes stay in 0-59 range
4. **Round-trip** — convert to minutes and back, verify equality

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/76)*
