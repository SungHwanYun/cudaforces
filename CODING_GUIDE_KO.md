# 📘 CUDA Online Judge 코딩 가이드

**CUDA 프로그래밍 학습을 위한 온라인 저지 플랫폼 사용 가이드**

---

## 📚 목차

1. [시스템 개요](#1-시스템-개요)
2. [CPU Transpiler란?](#2-cpu-transpiler란)
3. [코드 검증 (Validation)](#3-코드-검증-validation)
4. [지원 기능 목록](#4-지원-기능-목록)
5. [사용 가능한 라이브러리](#5-사용-가능한-라이브러리)
6. [사용 금지 항목](#6-사용-금지-항목)
7. [에러 코드 목록](#7-에러-코드-목록)
8. [코딩 가이드라인](#8-코딩-가이드라인)
9. [주의사항 및 제한사항](#9-주의사항-및-제한사항)
10. [자주 묻는 질문 (FAQ)](#10-자주-묻는-질문-faq)

---

## 1. 시스템 개요

### 1.1 CUDA Online Judge란?

CUDA Online Judge는 GPU 프로그래밍을 학습하기 위한 교육용 온라인 채점 시스템입니다. 
실제 GPU가 없어도 CUDA 코드를 작성하고 테스트할 수 있습니다.

### 1.2 채점 방식

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  CUDA 코드  │ ---> │ Transpiler  │ ---> │  C++ 코드   │ ---> │  CPU 실행   │
│  (제출)     │ 변환  │             │ 생성  │             │ 컴파일 │  & 채점    │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
```

- 제출된 CUDA 코드는 **CPU에서 실행 가능한 C++ 코드로 변환**됩니다
- OpenMP를 사용하여 GPU의 병렬 처리를 시뮬레이션합니다
- 출력 결과의 **정확성**만 평가합니다

### 1.3 채점 결과 종류

| 결과 | 설명 |
|------|------|
| ✅ **Accepted (AC)** | 정답 - 모든 테스트 케이스 통과 |
| ❌ **Wrong Answer (WA)** | 오답 - 출력이 정답과 다름 |
| ⚠️ **Compile Error (CE)** | 컴파일 에러 - 문법 오류 |
| ⏱️ **Time Limit Exceeded (TLE)** | 시간 초과 |
| 💾 **Memory Limit Exceeded (MLE)** | 메모리 초과 |
| 🚫 **Runtime Error (RE)** | 런타임 에러 - 세그폴트 등 |
| 🔒 **Validation Error (VE)** | 검증 실패 - CUDA 규칙 위반 |

---

## 2. CPU Transpiler란?

### 2.1 개념

**CPU Transpiler**는 CUDA 코드를 GPU 없이 CPU에서 실행할 수 있도록 변환하는 시스템입니다.

```cuda
// 원본 CUDA 코드
__global__ void add(int* a, int* b, int* c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// 커널 호출
add<<<1, 5>>>(d_a, d_b, d_c);
```

위 코드는 다음과 같이 변환됩니다:

```cpp
// 변환된 C++ 코드 (OpenMP 병렬화)
void add_impl(int threadIdx_x, ..., int* a, int* b, int* c) {
    struct { int x; } threadIdx = {threadIdx_x};
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// 커널 런치 → OpenMP 루프
#pragma omp parallel for
for (int tx = 0; tx < 5; tx++) {
    add_impl(tx, ..., d_a, d_b, d_c);
}
```

### 2.2 변환 방식

| CUDA 요소 | CPU 변환 방식 |
|-----------|--------------|
| `__global__` 함수 | 일반 C++ 함수 |
| `<<<blocks, threads>>>` | OpenMP 중첩 루프 |
| `threadIdx.x/y/z` | 함수 파라미터 |
| `blockIdx.x/y/z` | 함수 파라미터 |
| `blockDim.x/y/z` | 함수 파라미터 |
| `cudaMalloc` | `malloc` |
| `cudaMemcpy` | `memcpy` |
| `cudaFree` | `free` |
| `__shared__` | 블록별 독립 배열 |
| `atomicAdd` | `__atomic_fetch_add` |
| `__syncthreads()` | `#pragma omp barrier` |

### 2.3 ⚠️ 성능 벤치마킹 불가

> **중요**: 이 시스템은 **정확성 검증**만을 목적으로 합니다.

**성능 측정이 의미없는 이유:**

1. **실제 GPU와 다른 실행 환경**
   - GPU는 수천 개의 스레드를 동시에 실행
   - CPU 에뮬레이션은 순차적으로 시뮬레이션
   
2. **메모리 구조 차이**
   - GPU의 고속 메모리 계층 (L1/L2/Shared/Global)이 없음
   - 모든 메모리가 시스템 RAM으로 매핑

3. **시간 복잡도 차이**
   - GPU에서 O(1)인 병렬 연산이 CPU에서 O(n)이 될 수 있음

```cuda
// 이 코드의 GPU 성능 ≠ CPU 트랜스파일 성능
__global__ void matMul(float* A, float* B, float* C, int N) {
    // GPU: 모든 스레드 동시 실행
    // CPU: 스레드를 순차적으로 시뮬레이션
}
```

**권장 사항:**
- 알고리즘의 **정확성**만 검증하세요
- 실제 성능 최적화는 GPU가 있는 환경에서 테스트하세요
- `cudaEvent`로 측정한 시간은 의미가 없습니다 (항상 0 반환)

---

## 3. 코드 검증 (Validation)

제출된 코드는 다음 검증을 통과해야 합니다:

### 3.1 필수 요구사항

| 검증 항목 | 설명 | 에러 코드 |
|-----------|------|-----------|
| ✅ **커널 존재** | 최소 1개의 `__global__` 함수 필요 | E3001 |
| ✅ **의미있는 연산** | 커널이 실제 계산을 수행해야 함 | E3002 |
| ✅ **병렬처리 사용** | `threadIdx`, `blockIdx` 등 사용 필요 | E3003 |
| ✅ **GPU 메모리 사용** | `cudaMalloc`, `cudaMemcpy` 사용 필요 | E3004 |
| ✅ **커널 호출** | 정의된 커널을 `<<<>>>` 구문으로 호출 | E3001 |
| ❌ **금지 함수 미사용** | `qsort`, STL 함수 등 사용 불가 | E3005 |
| ❌ **금지 타입 미사용** | `std::vector`, `std::string` 등 사용 불가 | E3006 |

### 3.2 의미있는 커널 조건

다음 중 **하나 이상**을 만족해야 합니다:

```cuda
// ✅ 조건 1: 계산 수행
__global__ void kernel1(int* a, int* b, int* c) {
    c[i] = a[i] + b[i];  // 산술 연산
}

// ✅ 조건 2: 파라미터 사용
__global__ void kernel2(int* data, int n) {
    data[threadIdx.x] = n;  // 파라미터 접근
}

// ✅ 조건 3: 배열 접근
__global__ void kernel3(int* arr) {
    arr[threadIdx.x] = 1;  // 메모리 접근
}

// ✅ 조건 4: 출력 수행
__global__ void kernel4() {
    printf("Hello from GPU!\n");  // 출력
}
```

### 3.3 병렬처리 검증

커널 내에서 다음 빌트인 변수를 **하나 이상** 사용해야 합니다:

- `threadIdx.x`, `threadIdx.y`, `threadIdx.z`
- `blockIdx.x`, `blockIdx.y`, `blockIdx.z`
- `blockDim.x`, `blockDim.y`, `blockDim.z`
- `gridDim.x`, `gridDim.y`, `gridDim.z`

```cuda
// ❌ 검증 실패: 병렬처리 미사용
__global__ void bad_kernel(int* arr) {
    arr[0] = 1;  // 모든 스레드가 같은 작업
}

// ✅ 검증 통과: 병렬처리 사용
__global__ void good_kernel(int* arr) {
    int i = threadIdx.x;  // 스레드별 다른 인덱스
    arr[i] = i;
}
```

### 3.4 GPU 메모리 검증

main 함수에서 다음 함수를 **하나 이상** 사용해야 합니다:

- `cudaMalloc`
- `cudaMemcpy` (또는 `cudaMemcpyAsync`)
- `cudaMemset` (또는 `cudaMemsetAsync`)

```cuda
// ❌ 검증 실패: GPU 메모리 미사용
int main() {
    int arr[10];
    kernel<<<1, 10>>>(arr);  // 직접 호스트 메모리 전달
}

// ✅ 검증 통과: GPU 메모리 사용
int main() {
    int *d_arr;
    cudaMalloc(&d_arr, 10 * sizeof(int));  // GPU 메모리 할당
    cudaMemcpy(d_arr, arr, 10 * sizeof(int), cudaMemcpyHostToDevice);
    kernel<<<1, 10>>>(d_arr);
    cudaFree(d_arr);
}
```

---

## 4. 지원 기능 목록

### 4.1 함수 키워드

| 키워드 | 설명 | 지원 |
|--------|------|------|
| `__global__` | GPU에서 실행되는 커널 함수 | ✅ |
| `__device__` | GPU에서만 호출 가능한 함수 | ✅ |
| `__host__` | CPU에서 실행되는 함수 | ✅ |
| `__host__ __device__` | CPU/GPU 모두 호출 가능 | ✅ |

### 4.2 메모리 키워드

| 키워드 | 설명 | 지원 |
|--------|------|------|
| `__shared__` | 블록 내 공유 메모리 (정적) | ✅ |
| `extern __shared__` | 동적 공유 메모리 | ✅ |
| `__device__` | 디바이스 전역 변수 | ✅ |
| `__constant__` | 상수 메모리 | ✅ |

### 4.3 빌트인 변수

| 변수 | 설명 | 지원 |
|------|------|------|
| `threadIdx.x/y/z` | 블록 내 스레드 인덱스 | ✅ |
| `blockIdx.x/y/z` | 그리드 내 블록 인덱스 | ✅ |
| `blockDim.x/y/z` | 블록당 스레드 수 | ✅ |
| `gridDim.x/y/z` | 그리드 내 블록 수 | ✅ |
| `warpSize` | 워프 크기 (32) | ✅ |

### 4.4 메모리 관리 함수

| 함수 | 설명 | 지원 |
|------|------|------|
| `cudaMalloc` | GPU 메모리 할당 | ✅ |
| `cudaFree` | GPU 메모리 해제 | ✅ |
| `cudaMemcpy` | 메모리 복사 | ✅ |
| `cudaMemcpyAsync` | 비동기 메모리 복사 | ✅ |
| `cudaMemset` | 메모리 초기화 | ✅ |
| `cudaMemsetAsync` | 비동기 메모리 초기화 | ✅ |
| `cudaMemcpyToSymbol` | 심볼로 복사 | ✅ |
| `cudaMemcpyFromSymbol` | 심볼에서 복사 | ✅ |
| `cudaMemGetInfo` | 메모리 정보 조회 | ✅ |

### 4.5 Atomic 연산

| 함수 | 설명 | 지원 |
|------|------|------|
| `atomicAdd` | 원자적 덧셈 | ✅ |
| `atomicSub` | 원자적 뺄셈 | ✅ |
| `atomicExch` | 원자적 교환 | ✅ |
| `atomicMin` | 원자적 최솟값 | ✅ |
| `atomicMax` | 원자적 최댓값 | ✅ |
| `atomicInc` | 원자적 증가 (모듈러) | ✅ |
| `atomicDec` | 원자적 감소 (모듈러) | ✅ |
| `atomicCAS` | Compare-And-Swap | ✅ |
| `atomicAnd` | 원자적 AND | ✅ |
| `atomicOr` | 원자적 OR | ✅ |
| `atomicXor` | 원자적 XOR | ✅ |

### 4.6 동기화 함수

| 함수 | 설명 | 지원 |
|------|------|------|
| `__syncthreads()` | 블록 내 스레드 동기화 | ✅ |
| `__syncwarp()` | 워프 내 동기화 | ✅ |
| `cudaDeviceSynchronize()` | 디바이스 동기화 | ✅ |
| `cudaStreamSynchronize()` | 스트림 동기화 | ✅ |

### 4.7 Warp 연산

| 함수 | 설명 | 지원 |
|------|------|------|
| `__shfl_sync` | 워프 셔플 | ✅ |
| `__shfl_up_sync` | 업 셔플 | ✅ |
| `__shfl_down_sync` | 다운 셔플 | ✅ |
| `__shfl_xor_sync` | XOR 셔플 | ✅ |
| `__ballot_sync` | 워프 투표 | ✅ |
| `__all_sync` | 전체 참 검사 | ✅ |
| `__any_sync` | 일부 참 검사 | ✅ |
| `__activemask()` | 활성 스레드 마스크 | ✅ |

### 4.8 스트림 및 이벤트

| 함수 | 설명 | 지원 |
|------|------|------|
| `cudaStreamCreate` | 스트림 생성 | ✅ |
| `cudaStreamDestroy` | 스트림 제거 | ✅ |
| `cudaStreamSynchronize` | 스트림 동기화 | ✅ |
| `cudaEventCreate` | 이벤트 생성 | ✅ |
| `cudaEventRecord` | 이벤트 기록 | ✅ |
| `cudaEventSynchronize` | 이벤트 동기화 | ✅ |
| `cudaEventElapsedTime` | 경과 시간 (항상 0) | ⚠️ |

### 4.9 텍스처 메모리

| 함수 | 설명 | 지원 |
|------|------|------|
| `tex1D` | 1D 텍스처 읽기 | ✅ |
| `tex2D` | 2D 텍스처 읽기 | ✅ |
| `tex1Dfetch` | 1D 정수 좌표 | ✅ |
| `tex2Dfetch` | 2D 정수 좌표 | ✅ |

### 4.10 데이터 타입

| 타입 | 지원 |
|------|------|
| `int`, `unsigned int` | ✅ |
| `float`, `double` | ✅ |
| `char`, `unsigned char` | ✅ |
| `short`, `unsigned short` | ✅ |
| `long`, `unsigned long` | ✅ |
| `long long`, `unsigned long long` | ✅ |
| `size_t` | ✅ |
| `bool` | ✅ |
| `void` | ✅ |
| `dim3` | ✅ |
| 포인터 (`int*`, `float**`) | ✅ |
| 배열 (`int arr[N]`) | ✅ |
| 다차원 배열 | ✅ |
| `struct` | ✅ |
| `enum` | ✅ |
| `typedef` | ✅ |

### 4.11 연산자

| 카테고리 | 연산자 | 지원 |
|----------|--------|------|
| 산술 | `+`, `-`, `*`, `/`, `%` | ✅ |
| 비교 | `==`, `!=`, `<`, `>`, `<=`, `>=` | ✅ |
| 논리 | `&&`, `\|\|`, `!` | ✅ |
| 비트 | `&`, `\|`, `^`, `~`, `<<`, `>>` | ✅ |
| 대입 | `=`, `+=`, `-=`, `*=`, `/=`, `%=` | ✅ |
| 대입 (비트) | `&=`, `\|=`, `^=`, `<<=`, `>>=` | ✅ |
| 증감 | `++`, `--` (전위/후위) | ✅ |
| 삼항 | `? :` | ✅ |
| 포인터 | `*`, `&`, `->` | ✅ |
| sizeof | `sizeof(type)`, `sizeof(expr)` | ✅ |
| 캐스팅 | `(type)expr` | ✅ |

### 4.12 제어 구조

| 구조 | 지원 |
|------|------|
| `if-else` | ✅ |
| `for` 루프 | ✅ |
| `while` 루프 | ✅ |
| `do-while` 루프 | ✅ |
| `switch-case-default` | ✅ |
| `break` | ✅ |
| `continue` | ✅ |
| `return` | ✅ |

---

## 5. 사용 가능한 라이브러리

### 5.1 허용된 헤더 (stdio.h)

```c
// 입출력 함수
printf()     // 출력
scanf()      // 입력
fprintf()    // 파일 출력
fscanf()     // 파일 입력
fopen()      // 파일 열기
fclose()     // 파일 닫기
fread()      // 바이너리 읽기
fwrite()     // 바이너리 쓰기
fgets()      // 라인 읽기
fputs()      // 라인 쓰기
getchar()    // 문자 입력
putchar()    // 문자 출력
```

### 5.2 허용된 함수 (stdlib.h 일부)

```c
// 메모리 관리
malloc()     // 메모리 할당
calloc()     // 0 초기화 메모리 할당
realloc()    // 메모리 재할당
free()       // 메모리 해제

// 변환 함수
atoi()       // 문자열 → 정수
atof()       // 문자열 → 실수
atol()       // 문자열 → long
strtol()     // 문자열 → long (진법 지정)
strtod()     // 문자열 → double

// 난수
rand()       // 난수 생성
srand()      // 시드 설정

// 기타
abs()        // 절댓값
exit()       // 프로그램 종료
```

### 5.3 허용된 수학 함수 (math.h)

```c
// 기본 수학 함수
sin(), cos(), tan()      // 삼각함수
asin(), acos(), atan()   // 역삼각함수
sinh(), cosh(), tanh()   // 쌍곡선함수
exp(), log(), log10()    // 지수/로그
pow(), sqrt()            // 거듭제곱/제곱근
ceil(), floor(), round() // 올림/내림/반올림
fabs(), fmod()           // 절댓값/나머지
fmin(), fmax()           // 최솟값/최댓값
```

### 5.4 허용된 함수 (string.h / cstring)

```c
// 문자열 복사/연결
strcpy()     // 문자열 복사
strncpy()    // 문자열 복사 (n개 문자)
strcat()     // 문자열 연결
strncat()    // 문자열 연결 (n개 문자)

// 문자열 비교
strcmp()     // 문자열 비교
strncmp()    // 문자열 비교 (n개 문자)

// 문자열 검색
strlen()     // 문자열 길이
strchr()     // 문자 찾기 (처음)
strrchr()    // 문자 찾기 (마지막)
strstr()     // 부분 문자열 찾기
strpbrk()    // 문자 집합에서 찾기
strspn()     // 문자 집합 내 연속 길이
strcspn()    // 문자 집합 외 연속 길이
strtok()     // 토큰 분리

// 메모리 조작
memcpy()     // 메모리 복사
memmove()    // 메모리 이동 (중첩 안전)
memcmp()     // 메모리 비교
memset()     // 메모리 설정
memchr()     // 메모리에서 바이트 찾기
```

---

## 6. 사용 금지 항목

### 6.1 금지된 함수 목록

> ⚠️ 다음 함수들은 **직접 구현**해야 합니다.

#### stdlib.h 금지 함수
```c
qsort()      // ❌ 정렬 직접 구현 필요
bsearch()    // ❌ 이진 탐색 직접 구현 필요
```

#### STL algorithm 함수
```c
// 정렬
sort(), stable_sort(), partial_sort(), nth_element()

// 검색
find(), find_if(), find_first_of(), binary_search()
lower_bound(), upper_bound(), equal_range()

// 수정
copy(), fill(), transform(), replace(), swap()
reverse(), rotate(), shuffle(), unique(), remove()

// 집계
count(), count_if(), accumulate(), inner_product()
min(), max(), min_element(), max_element()

// 반복
for_each(), all_of(), any_of(), none_of()
```

#### STL 컨테이너 메서드
```c
push_back(), pop_back(), push_front(), pop_front()
emplace(), insert(), erase(), clear(), resize()
begin(), end(), front(), back(), at(), size()
```

### 6.2 금지된 타입 목록

> ⚠️ C++ STL 컨테이너는 사용할 수 없습니다. **C 스타일 배열과 포인터**를 사용하세요.

```cpp
// ❌ 금지된 STL 컨테이너
std::vector<T>         // → int arr[N] 또는 int* arr 사용
std::string            // → char arr[N] 또는 char* 사용
std::map<K,V>          // → 직접 구현 필요
std::unordered_map<K,V>
std::set<T>
std::unordered_set<T>
std::list<T>
std::deque<T>
std::queue<T>
std::stack<T>
std::priority_queue<T>
std::pair<T,U>
std::tuple<...>
std::array<T,N>
std::bitset<N>

// ❌ 금지된 동기화 타입
std::mutex
std::thread
std::atomic<T>

// ❌ 금지된 스마트 포인터
std::shared_ptr<T>
std::unique_ptr<T>
std::weak_ptr<T>
```

### 6.3 올바른 대안

```cuda
// ❌ 틀린 예 (vector 사용)
std::vector<int> arr(N);

// ✅ 올바른 예 (C 스타일 배열)
int arr[N];           // 고정 크기
int* arr = (int*)malloc(N * sizeof(int));  // 동적 할당

// ❌ 틀린 예 (string 사용)
std::string str = "hello";

// ✅ 올바른 예 (C 스타일 문자열)
char str[] = "hello";
char* str = "hello";

// ❌ 틀린 예 (sort 사용)
std::sort(arr, arr + n);

// ✅ 올바른 예 (직접 구현)
__device__ void bubbleSort(int* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}
```

---

## 7. 에러 코드 목록

### 7.1 E1xxx: 문법 에러 (SYNTAX_ERROR)

> 사용자가 수정해야 하는 CUDA C 문법 오류

| 코드 | 이름 | 설명 | 해결 방법 |
|------|------|------|-----------|
| E1001 | UNEXPECTED_TOKEN | 예상치 못한 토큰 | 문법 확인 |
| E1002 | MISSING_SEMICOLON | 세미콜론 누락 | `;` 추가 |
| E1003 | UNMATCHED_BRACKET | 괄호 불일치 | `{}`, `()`, `[]` 쌍 확인 |
| E1004 | INVALID_EXPRESSION | 잘못된 표현식 | 표현식 문법 확인 |
| E1005 | MISSING_TYPE | 타입 누락 | 변수/함수 타입 명시 |
| E1006 | INVALID_DECLARATION | 잘못된 선언 | 선언문 확인 |
| E1007 | EXPECTED_IDENTIFIER | 식별자 필요 | 변수/함수명 확인 |
| E1008 | INVALID_KERNEL_LAUNCH | 잘못된 커널 런치 | `<<<>>>` 구문 확인 |
| E1009 | INVALID_ARRAY_SIZE | 잘못된 배열 크기 | 배열 크기 확인 |
| E1010 | INVALID_OPERATOR | 잘못된 연산자 | 연산자 확인 |

### 7.2 E2xxx: 의미론 에러 (SEMANTIC_ERROR)

> CUDA 의미론 규칙 위반

| 코드 | 이름 | 설명 | 해결 방법 |
|------|------|------|-----------|
| E2001 | INVALID_MEMCPY_DIRECTION | cudaMemcpy 방향 오류 | Host/Device 포인터 확인 |
| E2002 | DEVICE_VAR_IN_HOST | 호스트에서 __device__ 접근 | cudaMemcpyToSymbol 사용 |
| E2003 | HOST_VAR_IN_DEVICE | 커널에서 호스트 변수 접근 | 파라미터로 전달 |
| E2004 | SHARED_MEMORY_IN_HOST | 호스트에서 __shared__ 사용 | 커널 내부에서만 사용 |
| E2005 | CONSTANT_WRITE_IN_KERNEL | 커널에서 __constant__ 쓰기 | __constant__는 읽기 전용 |
| E2006 | INVALID_MEMORY_ACCESS | 잘못된 메모리 접근 | 포인터 확인 |
| E2007 | HOST_FUNC_IN_KERNEL | 커널에서 호스트 함수 호출 | __device__ 함수 사용 |
| E2008 | HOST_VAR_ACCESS_IN_KERNEL | 커널에서 호스트 변수 접근 | __device__ 또는 파라미터 사용 |
| E2009 | DEVICE_FUNC_IN_HOST | 호스트에서 __device__ 함수 호출 | __host__ __device__ 추가 |
| E2010 | GLOBAL_FUNC_IN_KERNEL | 커널 내 커널 호출 시도 | 동적 병렬화 미지원 |

### 7.3 E3xxx: 검증 에러 (VALIDATION_ERROR)

> OJ 정책 위반 - 교육 목적상의 제한

| 코드 | 이름 | 설명 | 해결 방법 |
|------|------|------|-----------|
| E3001 | NO_KERNEL_FOUND | 커널 함수 없음/미호출 | `__global__` 함수 작성 및 호출 |
| E3002 | KERNEL_NOT_SIGNIFICANT | 커널이 의미없음 | 실제 연산 수행하는 코드 추가 |
| E3003 | NO_PARALLELISM | 병렬처리 미사용 | `threadIdx`, `blockIdx` 사용 |
| E3004 | NO_GPU_MEMORY_OPS | GPU 메모리 미사용 | `cudaMalloc`, `cudaMemcpy` 사용 |
| E3005 | FORBIDDEN_FUNCTION | 금지 함수 사용 | [금지 함수 목록](#61-금지된-함수-목록) 참조 |
| E3006 | FORBIDDEN_TYPE | 금지 타입 사용 | [금지 타입 목록](#62-금지된-타입-목록) 참조 |

### 7.4 E4xxx: 미지원 기능 (NOT_SUPPORTED)

> 트랜스파일러가 지원하지 않는 CUDA 기능

| 코드 | 이름 | 설명 | 대안 |
|------|------|------|------|
| E4001 | UNSUPPORTED_FEATURE | 일반 미지원 기능 | 문서 참조 |
| E4002 | COMPLEX_TEMPLATE | 복잡한 템플릿 | 단순화 필요 |
| E4003 | INLINE_PTX | 인라인 PTX 어셈블리 | C++ 코드로 대체 |
| E4005 | DYNAMIC_PARALLELISM | 동적 병렬처리 | 호스트에서 커널 호출 |
| E4006 | COOPERATIVE_GROUPS | 협력 그룹 | __syncthreads 사용 |
| E4008 | UNIFIED_MEMORY | 통합 메모리 | cudaMalloc + cudaMemcpy |

### 7.5 E5xxx: 내부 에러 (INTERNAL_ERROR)

> 시스템 내부 오류 (사용자 책임 아님)

| 코드 | 이름 | 설명 |
|------|------|------|
| E5001 | PARSER_INTERNAL | 파서 내부 오류 |
| E5002 | TRANSPILER_INTERNAL | 트랜스파일러 내부 오류 |
| E5003 | CODE_GEN_FAILED | 코드 생성 실패 |
| E5999 | UNKNOWN_INTERNAL | 알 수 없는 내부 오류 |

---

## 8. 코딩 가이드라인

### 8.1 기본 템플릿

```cuda
#include <stdio.h>

// 커널 함수 정의
__global__ void myKernel(int* input, int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 실제 연산 수행
        output[idx] = input[idx] * 2;
    }
}

int main() {
    int n;
    scanf("%d", &n);
    
    // 호스트 메모리 할당
    int* h_input = (int*)malloc(n * sizeof(int));
    int* h_output = (int*)malloc(n * sizeof(int));
    
    // 입력 읽기
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_input[i]);
    }
    
    // 디바이스 메모리 할당
    int *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    
    // 호스트 → 디바이스 복사
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // 커널 실행
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    myKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    
    // 디바이스 → 호스트 복사
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 결과 출력
    for (int i = 0; i < n; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");
    
    // 메모리 해제
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}
```

### 8.2 Atomic 연산 예제

```cuda
#include <stdio.h>

__global__ void sumKernel(int* arr, int n, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        atomicAdd(result, arr[idx]);  // 원자적 덧셈
    }
}

int main() {
    int n = 1000;
    int* h_arr = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) h_arr[i] = 1;
    
    int *d_arr, *d_result;
    int h_result = 0;
    
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);
    
    sumKernel<<<10, 100>>>(d_arr, n, d_result);
    
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Sum: %d\n", h_result);  // 1000
    
    cudaFree(d_arr);
    cudaFree(d_result);
    free(h_arr);
    
    return 0;
}
```

### 8.3 Shared Memory 예제

```cuda
#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void sharedMemSum(int* input, int* output, int n) {
    __shared__ int shared_data[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 공유 메모리에 로드
    shared_data[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    // 리덕션
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // 블록 결과 저장
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}
```

### 8.4 2D 그리드/블록 예제

```cuda
#include <stdio.h>

#define N 16
#define BLOCK_SIZE 4

__global__ void matrixAdd(int* A, int* B, int* C, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < width && col < width) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int size = N * N * sizeof(int);
    
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;
    
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);
    
    // 초기화
    for (int i = 0; i < N * N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // 2D 그리드 설정
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matrixAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // 결과 출력 (첫 번째 행만)
    for (int i = 0; i < N; i++) {
        printf("%d ", h_C[i]);
    }
    printf("\n");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
```

---

## 9. 주의사항 및 제한사항

### 9.1 동작 차이

| 항목 | GPU 동작 | CPU 에뮬레이션 |
|------|----------|----------------|
| 병렬 실행 | 동시 실행 | 순차 시뮬레이션 |
| `cudaEvent` 시간 | 실제 측정 | 항상 0 |
| 메모리 대역폭 | 고속 | 시스템 RAM 속도 |
| Warp 동기화 | 하드웨어 지원 | 소프트웨어 에뮬레이션 |
| 스트림 | 비동기 실행 | 동기 실행 |

### 9.2 Best Practices

1. **경계 검사 항상 수행**
   ```cuda
   if (idx < n) {
       // 안전한 접근
   }
   ```

2. **메모리 해제 잊지 않기**
   ```cuda
   cudaFree(d_ptr);
   free(h_ptr);
   ```

3. **적절한 블록 크기 사용**
   ```cuda
   // 일반적으로 128 ~ 512 권장
   int blockSize = 256;
   ```

4. **Shared Memory 크기 제한 고려**
   ```cuda
   // 블록당 48KB 제한 (실제 GPU)
   __shared__ float data[1024];  // 4KB
   ```

5. **Race Condition 방지**
   ```cuda
   atomicAdd(&sum, value);  // 동시 쓰기 시 atomic 사용
   ```

---

## 10. 자주 묻는 질문 (FAQ)

### Q1: 왜 `std::vector`를 사용할 수 없나요?

**A:** 교육 목적상, C 스타일 메모리 관리를 학습하도록 의도되었습니다. 
실제 CUDA 개발에서도 GPU 메모리는 C 스타일 포인터로 관리합니다.

```cuda
// ❌ 금지
std::vector<int> arr(N);

// ✅ 권장
int* arr = (int*)malloc(N * sizeof(int));
```

### Q2: 실행 시간이 느린 것 같은데, 코드가 비효율적인 건가요?

**A:** 아닙니다. CPU 트랜스파일러는 **정확성 검증용**이므로 실제 GPU 성능과 다릅니다.
실제 성능 테스트는 GPU 환경에서 해야 합니다.

### Q3: `cudaEventElapsedTime`이 항상 0을 반환해요.

**A:** 정상입니다. CPU 에뮬레이션에서는 모든 연산이 동기적으로 실행되므로 
경과 시간 측정이 의미가 없습니다.

### Q4: E3003 (NO_PARALLELISM) 에러가 발생해요.

**A:** 커널 내에서 `threadIdx`, `blockIdx` 등 병렬처리 변수를 사용해야 합니다.

```cuda
// ❌ 에러 발생
__global__ void kernel(int* arr) {
    arr[0] = 1;  // 모든 스레드가 같은 작업
}

// ✅ 해결
__global__ void kernel(int* arr) {
    int i = threadIdx.x;  // 스레드별 다른 인덱스
    arr[i] = i;
}
```

### Q5: 동적 공유 메모리는 어떻게 사용하나요?

**A:** `extern __shared__`와 커널 런치 시 세 번째 인자를 사용합니다.

```cuda
extern __shared__ int shared[];

__global__ void kernel(int* data, int n) {
    int tid = threadIdx.x;
    shared[tid] = data[tid];
    __syncthreads();
    // ...
}

int main() {
    // 세 번째 인자: 동적 shared memory 크기 (바이트)
    kernel<<<1, 256, 256 * sizeof(int)>>>(d_data, n);
}
```

### Q6: 왜 cudaMemcpy 방향 검증 에러(E2001)가 발생하나요?

**A:** `cudaMalloc`으로 할당한 포인터와 호스트 포인터의 방향이 맞지 않습니다.

```cuda
// ❌ 에러: d_arr은 Device인데 src로 사용
cudaMemcpy(h_arr, d_arr, size, cudaMemcpyHostToDevice);

// ✅ 올바름: d_arr은 src, DeviceToHost
cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
```

---

## 📞 도움말 및 지원

- **문의**: 📧 [ejpark29@gmail.com](mailto:ejpark29@gmail.com)

---

**버전**: 2.1.0  
**최종 업데이트**: 2025년 12월

---

**Happy CUDA Learning! 🚀**
