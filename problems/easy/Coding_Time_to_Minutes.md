# Coding Time to Minutes

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

A developer spent some time coding during the day. The coding duration is given in HH:MM format, where hours (HH) and minutes (MM) are both exactly 2 digits.

Convert the given time to total minutes.

### Input
A single line containing the coding time in HH:MM format.

**Constraints:**
- 00 ≤ HH ≤ 23
- 00 ≤ MM ≤ 59

### Output
Print the total coding time in minutes.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 02:30 | 150 |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 00:01 | 1 |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 10:00 | 600 |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void timeToMinutesKernel(int* hours, int* minutes, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = (*hours) * 60 + (*minutes);
    }
}

int main() {
    int hh, mm;
    scanf("%d:%d", &hh, &mm);
    
    // Device memory
    int *d_hours, *d_minutes, *d_result;
    cudaMalloc(&d_hours, sizeof(int));
    cudaMalloc(&d_minutes, sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    // Copy inputs to device
    cudaMemcpy(d_hours, &hh, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minutes, &mm, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    timeToMinutesKernel<<<1, 1>>>(d_hours, d_minutes, d_result);
    
    // Copy result back to host
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d\n", result);
    
    cudaFree(d_hours);
    cudaFree(d_minutes);
    cudaFree(d_result);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void timeToMinutesKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs time conversion on GPU |

---

## CUDA Concepts Covered

### 1. Parsing Formatted Input

Using `scanf` with format specifiers to parse HH:MM:

```cuda
int hh, mm;
scanf("%d:%d", &hh, &mm);

// Input: "02:30"
// hh = 2, mm = 30
// The ':' in the format string matches the ':' in input
```

### 2. Time Conversion Formula

Convert hours and minutes to total minutes:

```cuda
total_minutes = hours * 60 + minutes

// Examples:
// 02:30 → 2 * 60 + 30 = 150 minutes
// 00:01 → 0 * 60 + 1 = 1 minute
// 10:00 → 10 * 60 + 0 = 600 minutes
```

### 3. Visualization

```
Time: 02:30
       ↓
┌──────────────────────────────────────────┐
│  Hours: 2    Minutes: 30                 │
│     ↓            ↓                       │
│  2 × 60    +    30                       │
│     ↓            ↓                       │
│    120     +    30    =    150           │
└──────────────────────────────────────────┘
       ↓
Total: 150 minutes
```

### 4. Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  Input: "02:30" → hh = 2, mm = 30                        │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (HostToDevice)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                     DEVICE (GPU)                         │
│                                                          │
│   d_hours: [2]    d_minutes: [30]                        │
│          \          /                                    │
│           \        /                                     │
│      hours*60 + minutes                                  │
│       2*60  +   30  = 150                                │
│               │                                          │
│               ▼                                          │
│        d_result: [150]                                   │
│                                                          │
└──────────────────────────────────────────────────────────┘
                           │
              cudaMemcpy (DeviceToHost)
                           ↓
┌──────────────────────────────────────────────────────────┐
│                      HOST (CPU)                          │
│  result = 150 → printf("150\n")                          │
└──────────────────────────────────────────────────────────┘
```

### 5. Format String Parsing

The `scanf` format string `"%d:%d"` works as follows:

```
Format: "%d:%d"
Input:  "02:30"

%d   → reads "02" as integer 2
:    → matches literal ':'
%d   → reads "30" as integer 30
```

---

## Alternative Solutions

### Using Array

```cuda
__global__ void timeToMinutesKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // data[0] = hours, data[1] = minutes
        data[2] = data[0] * 60 + data[1];
    }
}

int main() {
    int hh, mm;
    scanf("%d:%d", &hh, &mm);
    
    int h_data[3] = {hh, mm, 0};
    
    int* d_data;
    cudaMalloc(&d_data, 3 * sizeof(int));
    cudaMemcpy(d_data, h_data, 3 * sizeof(int), cudaMemcpyHostToDevice);
    
    timeToMinutesKernel<<<1, 1>>>(d_data);
    
    cudaMemcpy(h_data, d_data, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d\n", h_data[2]);
    
    cudaFree(d_data);
    
    return 0;
}
```

### Direct Output in Kernel

```cuda
__global__ void timeToMinutesKernel(int* hours, int* minutes) {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("%d\n", (*hours) * 60 + (*minutes));
    }
}
```

### Reading as String and Parsing

```cuda
int main() {
    char time[6];  // "HH:MM" + null terminator
    scanf("%s", time);
    
    // Manual parsing
    int hh = (time[0] - '0') * 10 + (time[1] - '0');
    int mm = (time[3] - '0') * 10 + (time[4] - '0');
    
    // ... rest of the code
}
```

---

## Time Conversion Reference

| Hours | Minutes | Formula | Total |
|-------|---------|---------|-------|
| 0 | 1 | 0×60 + 1 | 1 |
| 1 | 0 | 1×60 + 0 | 60 |
| 2 | 30 | 2×60 + 30 | 150 |
| 10 | 0 | 10×60 + 0 | 600 |
| 23 | 59 | 23×60 + 59 | 1439 |

### Maximum Values

```
Max hours: 23
Max minutes: 59
Max total: 23 × 60 + 59 = 1439 minutes (just under 24 hours)
```

---

## Common Mistakes

### ❌ Forgetting the Colon in scanf
```cuda
scanf("%d%d", &hh, &mm);   // Wrong! Won't parse correctly
scanf("%d:%d", &hh, &mm);  // Correct - includes ':'
```

### ❌ Wrong Multiplication Factor
```cuda
*result = (*hours) * 100 + (*minutes);  // Wrong! Not base-100
*result = (*hours) * 60 + (*minutes);   // Correct - 60 minutes per hour
```

### ❌ Reversed Order
```cuda
*result = (*minutes) * 60 + (*hours);   // Wrong! Reversed
*result = (*hours) * 60 + (*minutes);   // Correct
```

### ❌ Integer Overflow (Not an issue here)
```cuda
// Max value: 23 * 60 + 59 = 1439
// Well within int range, no overflow concern
```

---

## Related Conversions

| Conversion | Formula |
|------------|---------|
| Hours → Minutes | `hours × 60` |
| Minutes → Hours | `minutes / 60` |
| Minutes → Remaining | `minutes % 60` |
| Seconds → Minutes | `seconds / 60` |
| Total → HH:MM | `total/60 : total%60` |

### Reverse Conversion Example

```cuda
// Total minutes back to HH:MM
int total = 150;
int hours = total / 60;    // 2
int minutes = total % 60;  // 30
// Result: "02:30"
```

---

## Key Takeaways

1. **Format string parsing** — `scanf("%d:%d", ...)` handles HH:MM format
2. **Time conversion** — hours × 60 + minutes = total minutes
3. **Leading zeros** — `scanf %d` handles "02" as 2 automatically
4. **Result range** — 0 to 1439 minutes (for 24-hour format)
5. **Common pattern** — base conversion (hours to minutes like base-60)

---

## Practice Exercises

1. Convert **minutes back to HH:MM** format
2. Calculate **duration between two times** (start and end)
3. Add **seconds** to the conversion (HH:MM:SS)
4. Handle **AM/PM** format conversion

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/75)*
