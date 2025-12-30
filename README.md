# 🚀 CUDA Online Judge

**Learn GPU Programming Without a GPU**

CUDA Online Judge (CUDAForces) is an educational platform for learning CUDA programming. Practice GPU programming concepts without expensive hardware through our innovative CPU Transpiler technology.

> 🌐 **Website**: [cudaforces.com](https://cudaforces.com)

---

## 📖 About This Repository

This repository provides **documentation, guides, and resources** for:

- 📚 **Learning CUDA** — Study guides and example problems
- 🔧 **Building Tools** — Public API documentation for developers
- 🎯 **Problem Solving** — Example solutions and coding patterns

---

## 📁 Repository Structure

```
cudaforces/
├── docs/
│   ├── QUICK_GUIDE.md              # Quick start guide
│   ├── COMPLETE_GUIDE_EN.md    # Comprehensive coding guide (English)
│   └── PUBLIC_API.md         # Public API documentation
│
├── problems/                  # Example problems with solutions
│   ├── easy/
│   ├── medium/
│   └── hard/
│
└── README.md
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [**QUICK_GUIDE.md**](docs/QUICK_GUIDE.md) | Quick introduction to CUDA Online Judge |
| [**COMPLETE_GUIDE.md**](docs/COMPLETE_GUIDE.md) | Complete coding guide with examples and FAQ |
| [**PUBLIC_API.md**](docs/PUBLIC_API.md) | API reference for building external tools |

---

## ⚙️ How It Works

CUDA Online Judge uses a **CPU Transpiler** that converts your CUDA code to run on CPU:

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  CUDA Code  │ ───► │ Transpiler  │ ───► │  C++ Code   │ ───► │ CPU Execute │
│   (.cu)     │      │  + Validate │      │  (OpenMP)   │      │  & Judge    │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
```

This allows you to:
- ✅ Learn CUDA syntax and concepts
- ✅ Verify algorithm correctness
- ✅ Practice without GPU hardware

> ⚠️ **Note**: Performance benchmarking is not available — the platform is for correctness verification only.

---

## 🔌 Public API

Build your own tools using our Public API!

**Base URL**: `https://cudaforces.com/api/v1`

### Available Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /problems` | List all public problems |
| `GET /problems/{id}` | Get problem metadata |
| `GET /problems/{id}/stats` | Get problem statistics |
| `GET /rankings` | Get user rankings |
| `GET /users/{username}` | Get user public profile |

### Quick Example

```python
import requests

response = requests.get("https://cudaforces.com/api/v1/problems", params={
    "difficulty": "Easy",
    "per_page": 10
})

for problem in response.json()["data"]["problems"]:
    print(f"#{problem['id']} - {problem['title']}")
```

📖 **Full documentation**: [PUBLIC_API.md](docs/PUBLIC_API.md)

---

## 💡 Tool Ideas

Here are some tools you can build with the API:

| Tool | Description | Main Endpoints |
|------|-------------|----------------|
| **Discord/Slack Bot** | Daily problem recommendations | `/problems` |
| **Progress Dashboard** | Visualize learning progress | `/users/{username}` |
| **GitHub Badge** | Display rank on your profile | `/users/{username}` |
| **VS Code Extension** | Browse problems in editor | `/problems` |
| **CLI Tool** | Command-line problem browser | All endpoints |

---

## 📝 Example Problem

### Print Hello World CUDA N Times (Easy)

Print "Hello World Cuda" exactly **n** times using CUDA parallelism.

```cuda
__global__ void helloKernel(int* dummy, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        printf("Hello World Cuda\n");
    }
}

int main() {
    int n;
    scanf("%d", &n);
    
    int* d_dummy;
    cudaMalloc(&d_dummy, sizeof(int));
    
    helloKernel<<<1, n>>>(d_dummy, n);
    
    cudaDeviceSynchronize();
    cudaFree(d_dummy);
    
    return 0;
}
```

📖 **More problems**: [problems/](problems/)

---

## ✅ Code Requirements

All submissions must follow these rules:

| Requirement | Description |
|-------------|-------------|
| ✅ Kernel Required | At least one `__global__` function |
| ✅ Use Parallelism | Must use `threadIdx`, `blockIdx`, etc. |
| ✅ GPU Memory | Must use `cudaMalloc`, `cudaMemcpy` |
| ❌ No STL | Cannot use `std::vector`, `std::string` |
| ❌ No Convenience Functions | Cannot use `qsort`, `strcpy`, etc. |

This ensures you learn authentic CUDA programming patterns!

---

## 🚀 Getting Started

1. **Visit** [cudaforces.com](https://cudaforces.com)
2. **Read** the [Coding Guide](docs/QUICK_GUIDE.md)
3. **Start** with Easy problems
4. **Build** tools using the [Public API](docs/PUBLIC_API.md)

---

## 🤝 Contributing

Contributions are welcome! You can help by:

- 📝 Improving documentation
- 🐛 Reporting issues
- 💡 Suggesting new features
- 🔧 Building and sharing tools

---

## 📧 Contact

- **Email**: ejpark29@gmail.com
- **Website**: [cudaforces.com](https://cudaforces.com)
- **Q&A Board**: [cudaforces.com/board](https://cudaforces.com/board)

---
