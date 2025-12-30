# Caesar Cipher Decryption

| Difficulty | Memory Limit | Time Limit | Author |
|------------|--------------|------------|--------|
| Easy | 128 MB | 1 s | MenOfPassion |

## Problem Description

John's lab developed a simple encryption algorithm that works with lowercase and uppercase letters.

Given a lowercase letter a and an integer k, the encryption process works as follows:

1. First, convert the lowercase letter a to uppercase to get b
2. Then, shift b by k positions in the alphabet to get the encrypted character c

If the resulting character would exceed 'Z' during the shift, it wraps around to 'A'. For example, if a = 'x' and k = 5, then b = 'X' and c = 'C'. If a = 'y' and k = 3, then b = 'Y' and c = 'B'.

Given the lowercase letter a and integer k, print the encrypted character c.

### Input
The first line contains a lowercase letter a and an integer k separated by a space.

**Constraints:**
- 1 ≤ k ≤ 10⁸

### Output
Print the encrypted character c on the first line.

### Examples

**Example 1**
| INPUT | OUTPUT |
|-------|--------|
| y 25 | X |

**Example 2**
| INPUT | OUTPUT |
|-------|--------|
| a 1000000 | O |

**Example 3**
| INPUT | OUTPUT |
|-------|--------|
| z 100000000 | V |

---

## Solution Code

> **Note**: CUDA Online Judge automatically removes all user-written `#include` statements and includes only the allowed libraries. You don't need to write any `#include` in your code.

```cuda
__global__ void caesarKernel(char* ch, int* k, char* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // Convert lowercase to uppercase
        char upper = *ch - 'a' + 'A';
        
        // Calculate position (0-25) and apply shift with modulo
        int pos = upper - 'A';
        int shift = *k % 26;
        int newPos = (pos + shift) % 26;
        
        *result = 'A' + newPos;
    }
}

int main() {
    char a;
    int k;
    scanf(" %c %d", &a, &k);
    
    // Device memory
    char *d_ch, *d_result;
    int *d_k;
    cudaMalloc(&d_ch, sizeof(char));
    cudaMalloc(&d_k, sizeof(int));
    cudaMalloc(&d_result, sizeof(char));
    
    // Copy inputs to device
    cudaMemcpy(d_ch, &a, sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    caesarKernel<<<1, 1>>>(d_ch, d_k, d_result);
    
    // Copy result back to host
    char result;
    cudaMemcpy(&result, d_result, sizeof(char), cudaMemcpyDeviceToHost);
    
    printf("%c\n", result);
    
    cudaFree(d_ch);
    cudaFree(d_k);
    cudaFree(d_result);
    
    return 0;
}
```

> 📚 New to CUDA OJ? Read the [CUDA Online Judge Guide](../../GUIDE.md) first.

### Why This Code Structure?

| Requirement | How We Satisfy It |
|-------------|-------------------|
| ✅ Kernel exists | `__global__ void caesarKernel()` |
| ✅ Uses parallelism | `threadIdx.x` for thread identification |
| ✅ Uses GPU memory | `cudaMalloc` / `cudaMemcpy` / `cudaFree` |
| ✅ Kernel called | `<<<1, 1>>>` launches the kernel |
| ✅ Meaningful computation | Character conversion and modular arithmetic |

---

## CUDA Concepts Covered

### 1. Character Case Conversion

Converting lowercase to uppercase using ASCII arithmetic:

```cuda
char upper = *ch - 'a' + 'A';

// 'a' has ASCII value 97
// 'A' has ASCII value 65
// Difference: 32

// Example: 'y' → 'Y'
// 'y' - 'a' = 24 (position in alphabet, 0-indexed)
// 24 + 'A' = 24 + 65 = 89 = 'Y'
```

### 2. Modular Arithmetic for Wrap-Around

The alphabet has 26 letters, so we use modulo 26:

```cuda
int shift = *k % 26;       // Reduce large k to 0-25 range
int newPos = (pos + shift) % 26;  // Wrap around if exceeds 25
```

### 3. Handling Large Numbers

With k up to 10⁸, direct addition could overflow or be inefficient:

```cuda
// Bad: Could be slow or overflow
for (int i = 0; i < k; i++) { pos++; pos %= 26; }

// Good: Use modulo directly
int shift = k % 26;  // 100,000,000 % 26 = 22
```

### 4. Step-by-Step Examples

**Example 1**: `y 25`
```
1. Input: a = 'y', k = 25
2. Uppercase: 'y' → 'Y'
3. Position: 'Y' - 'A' = 24
4. Shift: 25 % 26 = 25
5. New position: (24 + 25) % 26 = 49 % 26 = 23
6. Result: 'A' + 23 = 'X'
```

**Example 2**: `a 1000000`
```
1. Input: a = 'a', k = 1,000,000
2. Uppercase: 'a' → 'A'
3. Position: 'A' - 'A' = 0
4. Shift: 1,000,000 % 26 = 14
5. New position: (0 + 14) % 26 = 14
6. Result: 'A' + 14 = 'O'
```

**Example 3**: `z 100000000`
```
1. Input: a = 'z', k = 100,000,000
2. Uppercase: 'z' → 'Z'
3. Position: 'Z' - 'A' = 25
4. Shift: 100,000,000 % 26 = 22
5. New position: (25 + 22) % 26 = 47 % 26 = 21
6. Result: 'A' + 21 = 'V'
```

### 5. Alphabet Position Visualization

```
Position:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
Letter:    A  B  C  D  E  F  G  H  I  J  K  L  M  N  O  P  Q  R  S  T  U  V  W  X  Y  Z
                                                       ↑                          ↑
                                                    pos 14                      pos 24
                                                    = 'O'                       = 'Y'
```

---

## Alternative Solutions

### Combined Conversion and Shift

```cuda
__global__ void caesarKernel(char* ch, int* k, char* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // Direct calculation
        int pos = *ch - 'a';           // Position in lowercase alphabet
        int newPos = (pos + *k % 26) % 26;
        *result = 'A' + newPos;        // Convert to uppercase result
    }
}
```

### Using Array Indexing

```cuda
__global__ void caesarKernel(int* data, char* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        char ch = (char)data[0];
        int k = data[1];
        
        int pos = ch - 'a';
        int newPos = (pos + k % 26) % 26;
        *result = 'A' + newPos;
    }
}
```

---

## Common Mistakes

### ❌ Not Using Modulo for Large k
```cuda
// Wrong: Trying to add k directly when k is 10^8
int newPos = (pos + k) % 26;  // May cause issues with very large k
// Better:
int shift = k % 26;
int newPos = (pos + shift) % 26;
```

### ❌ Forgetting Case Conversion
```cuda
// Wrong: Output is lowercase instead of uppercase
*result = 'a' + newPos;
// Correct:
*result = 'A' + newPos;
```

### ❌ Wrong Position Calculation
```cuda
// Wrong: Using ASCII values directly
int newPos = (*ch + *k) % 26;
// Correct: Subtract base first
int pos = *ch - 'a';  // or *ch - 'A' if already uppercase
int newPos = (pos + shift) % 26;
```

### ❌ Missing Space in scanf
```cuda
scanf("%c %d", &a, &k);   // May fail due to whitespace issues
scanf(" %c %d", &a, &k);  // Space before %c handles leading whitespace
```

---

## ASCII Table Reference

| Character | ASCII | Position |
|-----------|-------|----------|
| 'A' | 65 | 0 |
| 'B' | 66 | 1 |
| ... | ... | ... |
| 'Z' | 90 | 25 |
| 'a' | 97 | 0 |
| 'b' | 98 | 1 |
| ... | ... | ... |
| 'z' | 122 | 25 |

**Key relationships:**
- `'A'` to `'Z'`: ASCII 65-90
- `'a'` to `'z'`: ASCII 97-122
- Difference: `'a' - 'A' = 32`

---

## Caesar Cipher Background

The Caesar cipher is one of the oldest known encryption techniques:

```
Original:   A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
Shift 3:    D E F G H I J K L M N O P Q R S T U V W X Y Z A B C
```

Named after Julius Caesar, who used it to protect military communications.

---

## Key Takeaways

1. **Modular arithmetic** — essential for wrap-around calculations
2. **Character arithmetic** — treat chars as integers for manipulation
3. **Large number handling** — reduce with modulo before operations
4. **ASCII relationships** — uppercase/lowercase differ by 32
5. **Position-based thinking** — convert to 0-25 range, then back to char

---

## Practice Exercises

1. Implement **decryption** (shift in opposite direction)
2. Handle **both uppercase and lowercase** input
3. Encrypt an **entire string** character by character
4. Implement **variable shift** per character position

---

*This problem is from [CUDA Online Judge](https://cudaforces.com/problem/26)*
