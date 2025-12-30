# CUDA Online Judge Public API v1

Public API for CUDA Online Judge. Designed to enable external developers to build independent tools and applications.

## Base URL

```
https://cudaforces.com/api/v1
```

## Authentication

The Public API currently requires no authentication. All endpoints return only public data.

## Rate Limiting

- **Limit**: 100 requests / minute / IP
- **Exceeded**: HTTP 429 response with `Retry-After` header

All responses include the following headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 60
```

## Response Format

All responses are in JSON format with the following structure:

### Success Response
```json
{
  "status": "success",
  "data": { ... },
  "timestamp": "2024-12-30T12:00:00Z"
}
```

### Error Response
```json
{
  "status": "error",
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable message"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `NOT_FOUND` | 404 | Resource not found |
| `ACCESS_DENIED` | 403 | Access denied |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Internal server error |

---

## Endpoints

### 1. API Information

```
GET /api/v1
```

Returns basic information about the API including version and available endpoints.

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "name": "CUDA Online Judge Public API",
    "version": "v1",
    "rate_limit": {
      "requests_per_minute": 100
    },
    "endpoints": {
      "GET /api/v1/problems": "List all public problems",
      "GET /api/v1/problems/{id}": "Get problem metadata",
      "GET /api/v1/problems/{id}/stats": "Get problem statistics",
      "GET /api/v1/rankings": "Get user rankings",
      "GET /api/v1/users/{username}": "Get user public profile"
    }
  }
}
```

---

### 2. List Problems

```
GET /api/v1/problems
```

Retrieves a list of all public problems.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | Page number |
| `per_page` | int | 20 | Items per page (max 100) |
| `difficulty` | string | - | Filter by difficulty (`Easy`, `Medium`, `Hard`) |
| `sort` | string | `id` | Sort order (`id`, `recent`) |

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "problems": [
      {
        "id": 1,
        "title": "Vector Addition",
        "difficulty": "Easy",
        "source": "CUDA Basics",
        "time_limit_ms": 5000,
        "memory_limit_mb": 1024,
        "submit_count": 342,
        "accepted_count": 267,
        "acceptance_rate": 78.07
      }
    ],
    "total": 50,
    "page": 1,
    "per_page": 20,
    "total_pages": 3
  },
  "timestamp": "2024-12-30T12:00:00Z"
}
```

---

### 3. Get Problem Details

```
GET /api/v1/problems/{problem_id}
```

Retrieves metadata for a specific problem.

> ⚠️ **Note**: For security reasons, problem content (description, input/output format, examples, hints) is not provided via API. To solve problems, please visit [cudaforces.com](https://cudaforces.com).

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `problem_id` | int | Problem ID |

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "id": 1,
    "title": "Vector Addition",
    "difficulty": "Easy",
    "source": "CUDA Basics",
    "author": "admin",
    "time_limit_ms": 5000,
    "memory_limit_mb": 1024,
    "submit_count": 342,
    "accepted_count": 267,
    "acceptance_rate": 78.07,
    "url": "https://cudaforces.com/problem/1"
  },
  "timestamp": "2024-12-30T12:00:00Z"
}
```

---

### 4. Get Problem Statistics

```
GET /api/v1/problems/{problem_id}/stats
```

Retrieves detailed statistics for a specific problem.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `problem_id` | int | Problem ID |

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "problem_id": 1,
    "total_submissions": 500,
    "result_distribution": {
      "Accepted": 267,
      "Wrong Answer": 150,
      "Time Limit Exceeded": 50,
      "Compile Error": 20,
      "Runtime Error": 13
    },
    "unique_users": 200,
    "solved_users": 150,
    "acceptance_rate": 53.4,
    "daily_submissions": [
      {"date": "2024-12-24", "count": 15},
      {"date": "2024-12-25", "count": 23}
    ]
  },
  "timestamp": "2024-12-30T12:00:00Z"
}
```

---

### 5. Get Rankings

```
GET /api/v1/rankings
```

Retrieves user rankings.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | Page number |
| `per_page` | int | 50 | Items per page (max 100) |

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "rankings": [
      {
        "rank": 1,
        "username": "cuda_master",
        "nickname": "CUDA Master",
        "solved_count": 48,
        "score": 245000,
        "tier": "Candidate Master",
        "tier_color": "#AA00AA"
      }
    ],
    "total": 1520,
    "page": 1,
    "per_page": 50,
    "total_pages": 31
  },
  "timestamp": "2024-12-30T12:00:00Z"
}
```

---

### 6. Get User Profile

```
GET /api/v1/users/{username}
```

Retrieves a user's public profile.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `username` | string | Username |

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "username": "cuda_master",
    "nickname": "CUDA Master",
    "school": "KAIST",
    "joined_at": "2024-06-15T09:00:00",
    "rank": 1,
    "score": 245000,
    "tier": "Candidate Master",
    "tier_color": "#AA00AA",
    "solved_count": 48,
    "solved_by_difficulty": {
      "Easy": 20,
      "Medium": 18,
      "Hard": 10
    },
    "total_submissions": 156,
    "social_links": {
      "github": "https://github.com/cuda_master",
      "linkedin": null,
      "twitter": null,
      "website": "https://cuda-master.dev"
    }
  },
  "timestamp": "2024-12-30T12:00:00Z"
}
```

---

## Code Examples

### Python

```python
import requests

BASE_URL = "https://cudaforces.com/api/v1"

# Get problem list
response = requests.get(f"{BASE_URL}/problems", params={
    "difficulty": "Easy",
    "per_page": 10
})
data = response.json()

if data["status"] == "success":
    for problem in data["data"]["problems"]:
        print(f"#{problem['id']} - {problem['title']} ({problem['difficulty']})")

```

### JavaScript (Node.js / Browser)

```javascript
const BASE_URL = 'https://cudaforces.com/api/v1';

// Using fetch (works in browser and Node.js 18+)
async function getProblems() {
  const response = await fetch(`${BASE_URL}/problems?difficulty=Easy&per_page=10`);
  const data = await response.json();
  
  if (data.status === 'success') {
    data.data.problems.forEach(problem => {
      console.log(`#${problem.id} - ${problem.title} (${problem.difficulty})`);
    });
  }
}

// Get user profile
async function getUserProfile(username) {
  const response = await fetch(`${BASE_URL}/users/${username}`);
  const data = await response.json();
  
  if (data.status === 'success') {
    console.log(`${data.data.nickname} - Rank #${data.data.rank}`);
    console.log(`Solved: ${data.data.solved_count} problems`);
  }
}

getProblems();
getUserProfile('cuda_master');
```

### cURL

```bash
# Get API info
curl "https://cudaforces.com/api/v1"

# List problems with filters
curl "https://cudaforces.com/api/v1/problems?difficulty=Easy&per_page=10"

# Get specific problem
curl "https://cudaforces.com/api/v1/problems/1"

# Get problem statistics
curl "https://cudaforces.com/api/v1/problems/1/stats"

# Get rankings
curl "https://cudaforces.com/api/v1/rankings?page=1&per_page=20"

# Get user profile
curl "https://cudaforces.com/api/v1/users/cuda_master"
```

---

## Tool Ideas

Here are some tools you can build using this API:

| Tool | Description | Main Endpoints |
|------|-------------|----------------|
| **Discord/Slack Bot** | Daily problem recommendations | `/problems` |
| **Learning Dashboard** | Visualize personal progress | `/users/{username}`, `/rankings` |
| **GitHub Badge** | Show rank/tier on README | `/users/{username}` |
| **VS Code Extension** | Browse problems in editor | `/problems`, `/problems/{id}` |
| **CLI Tool** | Command-line problem browser | All endpoints |

---

## Best Practices

1. **Cache responses** when possible to reduce API calls
2. **Handle rate limits** gracefully with exponential backoff
3. **Check response status** before accessing data
4. **Use pagination** for large datasets

```python
# Example: Handling rate limits
import time

def api_call_with_retry(url, max_retries=3):
    for i in range(max_retries):
        response = requests.get(url)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            continue
            
        return response.json()
    
    raise Exception("Max retries exceeded")
```

---

## Contact

- **Email**: ejpark29@gmail.com
- **Website**: https://cudaforces.com
- **Bug Reports**: https://cudaforces.com/board (Q&A Board)

---

## Changelog

### v1.0.0 (2024-12-30)
- Initial release
- Problem list/detail/statistics endpoints
- Rankings endpoint
- User profile endpoint
