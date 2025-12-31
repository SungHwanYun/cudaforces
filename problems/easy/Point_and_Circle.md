# Point and Circle

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

Given a circle O with center C = (a, b) and radius r, and a point A = (c, d), determine the position of point A relative to circle O.

The distance from point A to center C is calculated as:

$$dist = \sqrt{(c - a)^2 + (d - b)^2}$$

Based on this distance:
- If dist < r, point A is inside the circle → print "IN"
- If dist = r, point A is on the circle → print "ON"
- If dist > r, point A is outside the circle → print "OUT"

To avoid floating-point precision issues, compare the squared distance with r²:

$$dist^2 = (c - a)^2 + (d - b)^2$$

- If dist² < r², print "IN"
- If dist² = r², print "ON"
- If dist² > r², print "OUT"

### Input
The first line contains three integers a, b, and r — the center coordinates and radius of circle O.

The second line contains two integers c and d — the coordinates of point A.

**Constraints:**
- -100 ≤ a, b, c, d ≤ 100
- 1 ≤ r ≤ 100
- All values are integers.

### Output
Print "IN" if point A is inside the circle, "ON" if on the circle, or "OUT" if outside.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| 1 2 2<br>2 3 | IN |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| 0 0 5<br>3 4 | ON |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| 5 5 3<br>10 10 | OUT |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void pointCircleKernel(int* data, char* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int a = data[0];  // center x
        int b = data[1];  // center y
        int r = data[2];  // radius
        int c = data[3];  // point x
        int d = data[4];  // point y
        
        // Calculate squared distance (avoid sqrt for precision)
        int dx = c - a;
        int dy = d - b;
        int dist2 = dx * dx + dy * dy;
        int r2 = r * r;
        
        if (dist2 < r2) {
            result[0] = 'I'; result[1] = 'N'; result[2] = '\0';
        } else if (dist2 == r2) {
            result[0] = 'O'; result[1] = 'N'; result[2] = '\0';
        } else {
            result[0] = 'O'; result[1] = 'U'; result[2] = 'T'; result[3] = '\0';
        }
    }
}

int main() {
    int a, b, r, c, d;
    scanf("%d %d %d", &a, &b, &r);
    scanf("%d %d", &c, &d);
    
    int h_data[5] = {a, b, r, c, d};
    char h_result[4];
    
    // Device memory
    int *d_data;
    char *d_result;
    cudaMalloc(&d_data, 5 * sizeof(int));
    cudaMalloc(&d_result, 4 * sizeof(char));
    
    // Copy input to device
    cudaMemcpy(d_data, h_data, 5 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    pointCircleKernel<<<1, 1>>>(d_data, d_result);
    
    // Copy result back to host
    cudaMemcpy(h_result, d_result, 4 * sizeof(char), cudaMemcpyDeviceToHost);
    
    printf("%s\n", h_result);
    
    cudaFree(d_data);
    cudaFree(d_result);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void pointCircleKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Performs geometric calculation on GPU |

---

## CUDA Concepts Covered

### 1. Distance Formula

The Euclidean distance between two points:

$$dist = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

```cuda
// Traditional approach (has precision issues)
float dist = sqrtf((c-a)*(c-a) + (d-b)*(d-b));
if (dist < r) ...
```

### 2. Squared Distance Comparison

To avoid floating-point precision issues, compare squared values:

```cuda
// Better approach - integer arithmetic
int dist2 = dx * dx + dy * dy;
int r2 = r * r;

if (dist2 < r2) ...       // Inside
else if (dist2 == r2) ... // On
else ...                  // Outside
```

**Why this works:**
- If dist < r, then dist² < r² (both are positive)
- If dist = r, then dist² = r²
- If dist > r, then dist² > r²

### 3. Visualization

```
Example 1: Circle at (1,2), r=2, Point at (2,3)

        y
        │
      4 │     
        │   ╭───╮
      3 │  ╱  A ╲    A = (2,3)
        │ │  ·  │    C = (1,2)
      2 │ │  C  │    
        │  ╲   ╱     dist² = 1² + 1² = 2
      1 │   ╰─╯      r² = 4
        │            2 < 4 → IN
      0 └───────────── x
        0  1  2  3  4


Example 2: Circle at origin, r=5, Point at (3,4)

        y
        │
      5 │    ╭──────╮
        │   ╱        ╲
      4 │  │    A     │   A = (3,4)
        │  │  ·       │   dist² = 9 + 16 = 25
      3 │  │         │    r² = 25
        │  │    C    │    25 = 25 → ON
      2 │  │  (0,0)  │
        │   ╲        ╱
      0 │    ╰──────╯
        └───────────────── x
           0  2  4


Example 3: Circle at (5,5), r=3, Point at (10,10)

        y
       11│                    A
       10│                 ·  
        9│                    
        8│     ╭─╮            A = (10,10)
        7│    ╱   ╲           C = (5,5)
        6│   │  C  │          dist² = 25 + 25 = 50
        5│   │ (5,5)│         r² = 9
        4│    ╲   ╱           50 > 9 → OUT
        3│     ╰─╯
        └─────────────────── x
            4  5  6     10
```

### 4. Decision Logic

```
                    ┌────────────────────┐
                    │ Calculate dist²    │
                    │ dist² = dx² + dy²  │
                    └─────────┬──────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         dist² < r²      dist² == r²     dist² > r²
              │               │               │
              ▼               ▼               ▼
            "IN"            "ON"           "OUT"
```

### 5. Integer Overflow Consideration

With constraints |a|, |b|, |c|, |d| ≤ 100:
- Maximum dx or dy = 200
- Maximum dx² + dy² = 200² + 200² = 80,000
- Well within int range (2³¹ - 1 ≈ 2 billion)

---

## Alternative Solutions

### Using Float (Not Recommended)

```cuda
__global__ void pointCircleKernel(int* data, char* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int a = data[0], b = data[1], r = data[2];
        int c = data[3], d = data[4];
        
        float dx = (float)(c - a);
        float dy = (float)(d - b);
        float dist = sqrtf(dx * dx + dy * dy);
        
        // DANGER: Float comparison can be imprecise!
        if (dist < r) {
            // "IN"
        } else if (dist == r) {  // Might fail due to precision!
            // "ON"
        } else {
            // "OUT"
        }
    }
}
```

### Using Separate Variables

```cuda
__global__ void pointCircleKernel(int* center, int* radius, int* point, char* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int dx = point[0] - center[0];
        int dy = point[1] - center[1];
        int dist2 = dx * dx + dy * dy;
        int r2 = (*radius) * (*radius);
        
        if (dist2 < r2) {
            result[0] = 'I'; result[1] = 'N'; result[2] = '\0';
        } else if (dist2 == r2) {
            result[0] = 'O'; result[1] = 'N'; result[2] = '\0';
        } else {
            result[0] = 'O'; result[1] = 'U'; result[2] = 'T'; result[3] = '\0';
        }
    }
}
```

### Using printf in Kernel

```cuda
__global__ void pointCircleKernel(int* data) {
    int idx = threadIdx.x;
    if (idx == 0) {
        int dx = data[3] - data[0];
        int dy = data[4] - data[1];
        int dist2 = dx * dx + dy * dy;
        int r2 = data[2] * data[2];
        
        if (dist2 < r2) printf("IN\n");
        else if (dist2 == r2) printf("ON\n");
        else printf("OUT\n");
    }
}
```

---

## Why Squared Distance?

| Approach | Pros | Cons |
|----------|------|------|
| sqrt + float | Intuitive | Precision errors |
| Squared + int | Exact comparison | Overflow for large values |

**The Problem with sqrt:**
```
dist = sqrt(25) = 5.0000001 or 4.9999999
r = 5

dist == r might be false even when mathematically true!
```

**Squared comparison is exact:**
```
dist² = 25 (exactly)
r² = 25 (exactly)

dist² == r² is reliably true!
```

---

## Common Mistakes

### ❌ Using sqrt for Equality Check
```cuda
float dist = sqrtf(dx*dx + dy*dy);
if (dist == r) ...  // Unreliable!
```

### ❌ Comparing dist to r² 
```cuda
int dist2 = dx*dx + dy*dy;
if (dist2 == r) ...  // Wrong! Compare dist² to r², not r
if (dist2 == r*r) ...  // Correct
```

### ❌ Wrong Order of Subtraction
```cuda
int dx = a - c;  // Either order works for squared
int dy = b - d;  // (a-c)² = (c-a)²
```

### ❌ Integer Overflow (Not in This Problem)
```cuda
// For larger coordinates, use long long
long long dx = c - a;
long long dist2 = dx * dx + dy * dy;
```

---

## Geometric Relationships

| Condition | Meaning | Output |
|-----------|---------|--------|
| dist² < r² | Point closer than radius | IN |
| dist² = r² | Point exactly at radius | ON |
| dist² > r² | Point farther than radius | OUT |

---

## Key Takeaways

1. **Avoid sqrt** — use squared comparison for exact integer arithmetic
2. **Euclidean distance** — fundamental geometric concept
3. **Precision matters** — floating-point equality is unreliable
4. **Integer arithmetic** — exact and efficient for this problem
5. **Geometric classification** — inside/on/outside based on distance

---

## Practice Exercises

1. Check if a point is inside a **rectangle**
2. Determine if **two circles** intersect
3. Find the **closest point** on a circle to a given point
4. Check if a point is inside a **triangle**

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/134)*
