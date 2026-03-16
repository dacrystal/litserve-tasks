# litserve-tasks

Async task processing for [LitServe](https://github.com/Lightning-AI/LitServe) via `TaskSpec` — a `LitSpec` subclass.

Clients submit long-running inference jobs and receive a `task_id` immediately, then poll for status and fetch the result when ready. Designed to mirror LitServe's internal layout for easy future integration as a PR.

Related issues: [#405](https://github.com/Lightning-AI/LitServe/issues/405), [#348](https://github.com/Lightning-AI/LitServe/issues/348)

---

## Installation

```bash
pip install -e .
```

---

## Usage

### 1. Build your `LitAPI` and attach `TaskSpec`

```python
import litserve as ls
from pydantic import BaseModel
from litserve_tasks import TaskSpec

class MyRequest(BaseModel):
    input: int

class MyAPI(ls.LitAPI):
    def setup(self, device):
        self.model = lambda x: x ** 2

    def decode_request(self, request: MyRequest):
        return request.input

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output):
        return {"output": output}

api = MyAPI(spec=TaskSpec())
server = ls.LitServer(api, accelerator="cpu")
server.run(port=8000)
```

The server generates a `uuid4` task ID automatically. To let clients supply their own ID, add a `TaskIDField` to your request model:

```python
from litserve_tasks import TaskIDField

class MyRequest(BaseModel):
    task_id: TaskIDField   # optional — client-provided or auto-generated
    input: int
```

### 2. Custom base path

```python
api = MyAPI(api_path="/jobs", spec=TaskSpec())
# Endpoints become: POST /jobs, GET /jobs/{task_id}, etc.
```

---

## HTTP Endpoints

All paths relative to `api_path` (default: `/tasks`).

| Method   | Path                        | Description                         | Response                                          |
|----------|-----------------------------|-------------------------------------|---------------------------------------------------|
| `POST`   | `/tasks`                    | Submit task, return immediately     | `{"task_id": "..."}`                              |
| `GET`    | `/tasks/{task_id}`          | Poll task status                    | `{"status": "processing" \| "completed"}`         |
| `GET`    | `/tasks/{task_id}/result`   | Fetch result (202 if still running) | result payload or `HTTP 202`                      |
| `DELETE` | `/tasks/{task_id}`          | Explicit cleanup, free memory       | `{"detail": "Task {task_id} marked as done"}`     |

---

## Example Session

```bash
# Submit a task
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"input": 4}'
# → {"task_id": "3f2e1d..."}

# Poll status
curl http://localhost:8000/tasks/3f2e1d...
# → {"status": "processing"}
# → {"status": "completed"}

# Fetch result
curl http://localhost:8000/tasks/3f2e1d.../result
# → {"output": 16}

# Clean up
curl -X DELETE http://localhost:8000/tasks/3f2e1d...
# → {"detail": "Task 3f2e1d... marked as done"}
```

---

## Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Full suite
pytest tests/
```

---


## Future Work

- **TTL-based expiry** — background cleanup of completed tasks older than N seconds
- **Webhook notifications** — push result to a client-provided callback URL on completion instead of polling
- **Progress feedback** — stream intermediate progress updates via `yield` before the final result is ready
- **Task cancellation**
