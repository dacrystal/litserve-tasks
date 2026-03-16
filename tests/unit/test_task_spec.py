import asyncio
import queue
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from pydantic import BaseModel

import litserve as ls
from litserve.utils import LitAPIStatus, ResponseBufferItem
from litserve_tasks import TaskIDField, TaskSpec
from litserve_tasks.test_examples.simple_task_example import SimpleLitAPI, SimpleFormLitAPI


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_spec_with_api(api):
    spec = api.spec
    spec.pre_setup(api)
    return spec


def make_response_buffer_item(done: bool, response=None, status=LitAPIStatus.OK):
    event = asyncio.Event()
    if done:
        event.set()
    item = ResponseBufferItem(event=event)
    if done and response is not None:
        item.response = (response, status)
    return item


# ── TaskSpec instantiation ────────────────────────────────────────────────────

def test_default_api_path():
    spec = TaskSpec()
    assert spec.api_path == "/tasks"


def test_custom_api_path_via_lit_api():
    api = SimpleLitAPI(api_path="/jobs", spec=TaskSpec())
    spec = make_spec_with_api(api)
    assert spec.api_path == "/jobs"


def test_default_api_path_unchanged_when_not_overridden():
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    assert spec.api_path == "/tasks"


def test_pre_setup_registers_four_endpoints():
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    paths = [(path, methods) for path, _, methods in spec.endpoints]
    assert ("/tasks", ["POST"]) in paths
    assert ("/tasks/{task_id}", ["GET"]) in paths
    assert ("/tasks/{task_id}", ["DELETE"]) in paths
    assert ("/tasks/{task_id}/result", ["GET"]) in paths


def test_endpoints_use_custom_api_path():
    api = SimpleLitAPI(api_path="/jobs", spec=TaskSpec())
    spec = make_spec_with_api(api)
    paths = [path for path, _, _ in spec.endpoints]
    assert "/jobs" in paths
    assert "/tasks" not in paths


# ── TaskIDField extraction ────────────────────────────────────────────────────

def test_task_id_field_found_in_plain_model():
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    assert spec._task_id_field == "task_id"


def test_task_id_field_none_when_no_field():
    class NoIDRequest(BaseModel):
        input: int

    class NoIDAPI(ls.LitAPI):
        def setup(self, device): pass
        def decode_request(self, request: NoIDRequest): return request.input
        def predict(self, x): return x
        def encode_response(self, output): return output

    api = NoIDAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    assert spec._task_id_field is None


def test_task_id_field_found_in_annotated_form_model():
    """get_task_id_field() unwraps Annotated[Model, Form()] before inspecting fields."""
    api = SimpleFormLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    assert spec._task_id_field == "task_id"


def test_task_id_field_found_in_optional_model():
    """get_task_id_field() unwraps Optional[Model] before inspecting fields."""
    from typing import Optional

    class OptionalRequest(BaseModel):
        task_id: TaskIDField
        input: int

    class OptionalAPI(ls.LitAPI):
        def setup(self, device): pass
        def decode_request(self, request: Optional[OptionalRequest]): return request.input
        def predict(self, x): return x
        def encode_response(self, output): return output

    api = OptionalAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    assert spec._task_id_field == "task_id"


def test_get_task_id_uses_field_value():
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)

    from litserve_tasks.test_examples.simple_task_example import SimpleRequest
    req = SimpleRequest(task_id="my-custom-id", input=5)
    assert spec.get_task_id(req) == "my-custom-id"


def test_get_task_id_generates_uuid_when_no_field():
    class NoIDRequest(BaseModel):
        input: int

    class NoIDAPI(ls.LitAPI):
        def setup(self, device): pass
        def decode_request(self, request: NoIDRequest): return request.input
        def predict(self, x): return x
        def encode_response(self, output): return output

    api = NoIDAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    req = NoIDRequest(input=5)
    task_id = spec.get_task_id(req)
    import uuid
    uuid.UUID(task_id)  # raises ValueError if not valid uuid


# ── get_status ────────────────────────────────────────────────────────────────

def test_get_status_returns_processing_when_event_not_set():
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    spec.response_buffer = {"abc": make_response_buffer_item(done=False)}

    result = spec.get_status("abc")
    assert result == {"status": "processing"}


def test_get_status_returns_completed_when_event_set():
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    spec.response_buffer = {"abc": make_response_buffer_item(done=True, response={"output": 4})}

    result = spec.get_status("abc")
    assert result == {"status": "completed"}


def test_get_status_404_when_unknown_task():
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    spec.response_buffer = {}

    with pytest.raises(HTTPException) as exc_info:
        spec.get_status("nonexistent")
    assert exc_info.value.status_code == 404


# ── get_result ────────────────────────────────────────────────────────────────

def test_get_result_202_when_not_done():
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    spec.response_buffer = {"abc": make_response_buffer_item(done=False)}

    with pytest.raises(HTTPException) as exc_info:
        spec.get_result("abc")
    assert exc_info.value.status_code == 202


def test_get_result_returns_response_when_done():
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    spec.response_buffer = {
        "abc": make_response_buffer_item(done=True, response={"output": 16})
    }

    result = spec.get_result("abc")
    assert result == {"output": 16}


def test_get_result_404_when_unknown():
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    spec.response_buffer = {}

    with pytest.raises(HTTPException) as exc_info:
        spec.get_result("nonexistent")
    assert exc_info.value.status_code == 404


def test_get_result_500_on_error_status():
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    spec.response_buffer = {
        "abc": make_response_buffer_item(done=True, response="something went wrong", status=LitAPIStatus.ERROR)
    }

    with pytest.raises(HTTPException) as exc_info:
        spec.get_result("abc")
    assert exc_info.value.status_code == 500
    assert "something went wrong" in exc_info.value.detail  # str(response) preserved


def test_get_result_reraises_http_exception_from_worker():
    """Worker errors that are HTTPException subclasses must propagate as-is."""
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    worker_exc = HTTPException(status_code=422, detail="unprocessable entity")
    spec.response_buffer = {
        "abc": make_response_buffer_item(done=True, response=worker_exc, status=LitAPIStatus.ERROR)
    }

    with pytest.raises(HTTPException) as exc_info:
        spec.get_result("abc")
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "unprocessable entity"


def test_get_result_does_not_pop_from_buffer():
    """Result must remain in buffer until DELETE is called."""
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    spec.response_buffer = {
        "abc": make_response_buffer_item(done=True, response={"output": 9})
    }

    spec.get_result("abc")
    assert "abc" in spec.response_buffer  # still present


# ── task_done ─────────────────────────────────────────────────────────────────

def test_task_done_404_when_unknown():
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    spec.response_buffer = {}

    with pytest.raises(HTTPException) as exc_info:
        spec.task_done("nonexistent")
    assert exc_info.value.status_code == 404


def test_task_done_409_when_still_processing():
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    spec.response_buffer = {"abc": make_response_buffer_item(done=False)}

    with pytest.raises(HTTPException) as exc_info:
        spec.task_done("abc")
    assert exc_info.value.status_code == 409


def test_task_done_removes_from_buffer():
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    spec.response_buffer = {
        "abc": make_response_buffer_item(done=True, response={"output": 4})
    }
    mock_server = MagicMock()
    spec._server = mock_server

    result = spec.task_done("abc")
    assert result == {"detail": "Task 'abc' marked as done"}
    assert "abc" not in spec.response_buffer


def test_task_done_triggers_on_response_callback():
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    spec.response_buffer = {
        "abc": make_response_buffer_item(done=True, response={"output": 4})
    }
    mock_server = MagicMock()
    spec._server = mock_server

    spec.task_done("abc")
    mock_server._callback_runner.trigger_event.assert_called_once()


# ── _submit_task ──────────────────────────────────────────────────────────────

async def test_submit_task_rejects_duplicate_task_id():
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    spec.response_buffer = {"existing-id": make_response_buffer_item(done=False)}

    from litserve_tasks.test_examples.simple_task_example import SimpleRequest
    req = SimpleRequest(task_id="existing-id", input=5)

    with pytest.raises(HTTPException) as exc_info:
        await spec._submit_task(req, type(req))
    assert exc_info.value.status_code == 409


async def test_submit_task_cleans_buffer_on_queue_full():
    api = SimpleLitAPI(spec=TaskSpec())
    spec = make_spec_with_api(api)
    spec.response_buffer = {}

    mock_server = MagicMock()
    spec._server = mock_server

    with patch.object(spec, "_submit_request", side_effect=queue.Full):
        from litserve_tasks.test_examples.simple_task_example import SimpleRequest
        req = SimpleRequest(task_id="my-id", input=5)

        with pytest.raises(HTTPException) as exc_info:
            await spec._submit_task(req, type(req))

        assert exc_info.value.status_code == 503
        # Buffer must be cleaned up — not leave an orphaned entry
        assert "my-id" not in spec.response_buffer
