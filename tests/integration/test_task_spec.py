import asyncio
import time

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

import litserve as ls
from litserve.utils import wrap_litserve_start
from litserve_tasks import TaskSpec
from litserve_tasks.test_examples.simple_task_example import SimpleLitAPI, SimpleFormLitAPI


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def server():
    api = SimpleLitAPI(spec=TaskSpec())
    server = ls.LitServer(api, accelerator="cpu", devices=1, timeout=30)
    with wrap_litserve_start(server) as s:
        yield s


@pytest.fixture
def form_server():
    api = SimpleFormLitAPI(spec=TaskSpec())
    server = ls.LitServer(api, accelerator="cpu", devices=1, timeout=30)
    with wrap_litserve_start(server) as s:
        yield s


@pytest.fixture
async def client(server):
    async with (
        LifespanManager(server.app) as manager,
        AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
    ):
        yield ac


@pytest.fixture
async def form_client(form_server):
    async with (
        LifespanManager(form_server.app) as manager,
        AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
    ):
        yield ac


# ── Helpers ───────────────────────────────────────────────────────────────────

async def poll_until_complete(client, task_id, timeout=10.0, interval=0.1):
    """Poll GET /tasks/{task_id} until status is 'completed'."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = await client.get(f"/tasks/{task_id}")
        assert r.status_code == 200, f"Unexpected {r.status_code} polling {task_id}: {r.text}"
        if r.json()["status"] == "completed":
            return
        await asyncio.sleep(interval)
    raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")


# ── Full lifecycle ─────────────────────────────────────────────────────────────

async def test_full_lifecycle_json(client):
    """Submit → poll → result → delete."""
    # Submit
    r = await client.post("/tasks", json={"input": 4})
    assert r.status_code == 200
    task_id = r.json()["task_id"]
    assert task_id

    # Poll until done
    await poll_until_complete(client, task_id)

    # Fetch result
    r = await client.get(f"/tasks/{task_id}/result")
    assert r.status_code == 200
    assert r.json() == {"output": 16}

    # Delete
    r = await client.delete(f"/tasks/{task_id}")
    assert r.status_code == 200
    assert "marked as done" in r.json()["detail"]

    # Confirm deleted
    r = await client.get(f"/tasks/{task_id}")
    assert r.status_code == 404


async def test_custom_task_id(client):
    """Client-provided task_id via TaskIDField is preserved."""
    r = await client.post("/tasks", json={"task_id": "my-task-123", "input": 3})
    assert r.status_code == 200
    assert r.json()["task_id"] == "my-task-123"

    await poll_until_complete(client, "my-task-123")

    r = await client.get("/tasks/my-task-123/result")
    assert r.status_code == 200
    assert r.json() == {"output": 9}

    await client.delete("/tasks/my-task-123")


async def test_result_readable_multiple_times(client):
    """GET /result does not pop — can be called multiple times before DELETE."""
    r = await client.post("/tasks", json={"input": 5})
    task_id = r.json()["task_id"]

    await poll_until_complete(client, task_id)

    r1 = await client.get(f"/tasks/{task_id}/result")
    r2 = await client.get(f"/tasks/{task_id}/result")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json() == r2.json()

    await client.delete(f"/tasks/{task_id}")


async def test_result_returns_200_after_completion(client):
    """GET /result returns 200 once the task completes."""
    r = await client.post("/tasks", json={"input": 2})
    task_id = r.json()["task_id"]

    await poll_until_complete(client, task_id)
    r = await client.get(f"/tasks/{task_id}/result")
    assert r.status_code == 200
    assert r.json() == {"output": 4}  # 2**2
    await client.delete(f"/tasks/{task_id}")


async def test_duplicate_task_id_rejected(client):
    """Submitting the same task_id twice returns 409."""
    r = await client.post("/tasks", json={"task_id": "dup-task", "input": 2})
    assert r.status_code == 200

    r = await client.post("/tasks", json={"task_id": "dup-task", "input": 3})
    assert r.status_code == 409
    assert "dup-task" in r.json()["detail"]  # error mentions the conflicting task_id

    # Clean up the first task
    await poll_until_complete(client, "dup-task")
    await client.delete("/tasks/dup-task")


async def test_delete_in_flight_returns_409(client):
    """DELETE on a still-processing task returns 409."""
    r = await client.post("/tasks", json={"input": 6})
    task_id = r.json()["task_id"]

    # Try to delete immediately (before completion)
    r = await client.delete(f"/tasks/{task_id}")
    # May be 409 (still processing) or 200 (already done — fast worker)
    assert r.status_code in (200, 409)

    if r.status_code == 409:
        # Task still in flight — wait for it then clean up
        await poll_until_complete(client, task_id)
        await client.delete(f"/tasks/{task_id}")
    # If 200: task was already done and deleted — nothing more to clean up


async def test_unknown_task_returns_404(client):
    """All endpoints return 404 for unknown task_id."""
    r = await client.get("/tasks/nonexistent")
    assert r.status_code == 404

    r = await client.get("/tasks/nonexistent/result")
    assert r.status_code == 404

    r = await client.delete("/tasks/nonexistent")
    assert r.status_code == 404


async def test_form_submit_lifecycle(form_client):
    """Submit via multipart/form-data (with a file part) → poll → result → delete."""
    r = await form_client.post(
        "/tasks",
        data={"task_id": "form-task-1", "input": "4", "description": "test upload"},
        files={"file": ("test.txt", b"hello world", "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["task_id"] == "form-task-1"  # client-provided task_id preserved

    await poll_until_complete(form_client, "form-task-1")

    r = await form_client.get("/tasks/form-task-1/result")
    assert r.status_code == 200
    assert r.json() == {"output": 16}  # 4**2

    r = await form_client.delete("/tasks/form-task-1")
    assert r.status_code == 200


async def test_concurrent_tasks(client):
    """Multiple tasks can be in flight simultaneously."""
    inputs = list(range(5))

    # Submit concurrently with explicit task_ids to track input→output mapping
    responses = await asyncio.gather(*[
        client.post("/tasks", json={"task_id": f"concurrent-{i}", "input": i})
        for i in inputs
    ])
    for r in responses:
        assert r.status_code == 200

    task_ids = [f"concurrent-{i}" for i in inputs]
    assert len(set(task_ids)) == 5  # all unique

    # Wait for all to complete
    await asyncio.gather(*[poll_until_complete(client, tid) for tid in task_ids])

    # Verify results — task_id encodes input so ordering is explicit
    for i in inputs:
        r = await client.get(f"/tasks/concurrent-{i}/result")
        assert r.status_code == 200
        assert r.json() == {"output": i**2}
        await client.delete(f"/tasks/concurrent-{i}")
