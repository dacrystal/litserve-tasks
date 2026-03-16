import asyncio
import inspect
import logging
import queue
import time
import uuid
from typing import Annotated, Optional, Union, get_args, get_origin

from fastapi import HTTPException, Request, Response
from pydantic import BaseModel, Field

from litserve.callbacks.base import EventTypes
from litserve.constants import _DEFAULT_LIT_API_PATH
from litserve.specs.base import LitSpec
from litserve.utils import LitAPIStatus, ResponseBufferItem

logger = logging.getLogger(__name__)

# Annotated type alias — use as a field type in Pydantic request models.
# The server reads the task_id from this field; if absent, generates a uuid4.
TaskIDField = Annotated[
    str,
    Field(
        default_factory=lambda: str(uuid.uuid4()),
        json_schema_extra={"TASK_ID": True},
    ),
]


class TaskSpec(LitSpec):
    """LitServe spec for async/polling task processing.

    Submit a task and get a task_id immediately. Poll /tasks/{task_id} for
    status, fetch the result from /tasks/{task_id}/result, and delete with
    DELETE /tasks/{task_id} when done.
    """

    def __init__(self):
        super().__init__()
        self.api_path = "/tasks"
        self._request_type = None
        self._response_type = None
        self._task_id_field: Optional[str] = None

    def pre_setup(self, lit_api):
        decode_sig = inspect.signature(lit_api.decode_request)
        encode_sig = inspect.signature(lit_api.encode_response)

        request_type = decode_sig.parameters["request"].annotation
        if request_type is decode_sig.empty:
            request_type = Request

        response_type = encode_sig.return_annotation
        if response_type is encode_sig.empty:
            response_type = Response

        self._request_type = request_type
        self._response_type = response_type

        # Override api_path if LitAPI was given a custom path
        if lit_api._api_path and lit_api._api_path not in (_DEFAULT_LIT_API_PATH, self.api_path):
            self.api_path = lit_api._api_path

        # Patch submit_task's type annotations so FastAPI builds the right schema.
        self.submit_task.__func__.__annotations__ = self.submit_task.__func__.__annotations__ | {
            "request": request_type,
            "return": response_type,
        }

        self._task_id_field = self.get_task_id_field()

        # Register endpoints using self.api_path
        self.add_endpoint(self.api_path, self.submit_task, ["POST"])
        self.add_endpoint(self.api_path + "/{task_id}", self.get_status, ["GET"])
        self.add_endpoint(self.api_path + "/{task_id}", self.task_done, ["DELETE"])
        self.add_endpoint(self.api_path + "/{task_id}/result", self.get_result, ["GET"])

    def decode_request(self, request, context_kwargs):
        return request

    def encode_response(self, output, context_kwargs):
        return output

    def get_task_id_field(self) -> Optional[str]:
        """Return the name of the TaskIDField field in the request model, or None."""
        request_type = self._request_type

        # Unwrap Annotated[T, ...]
        origin = get_origin(request_type)
        if origin is Annotated:
            request_type = get_args(request_type)[0]

        # Unwrap Optional[T] / Union[T, None]
        origin = get_origin(request_type)
        if origin is Union:
            for arg in get_args(request_type):
                if arg is not type(None):
                    request_type = arg
                    break

        try:
            if isinstance(request_type, type) and issubclass(request_type, BaseModel):
                for name, field in request_type.model_fields.items():
                    if (field.json_schema_extra or {}).get("TASK_ID"):
                        return name
        except TypeError:
            pass

        return None

    def get_task_id(self, request) -> str:
        """Extract task_id from request's TaskIDField, or generate a uuid4."""
        if self._task_id_field is not None:
            return getattr(request, self._task_id_field)
        return str(uuid.uuid4())

    async def submit_task(self, request):
        return await self._submit_task(request, self._request_type)

    async def _submit_task(self, request, request_type):
        try:
            payload = await self._prepare_request(request, request_type)
            uid = self.get_task_id(payload)

            # Reject duplicate task IDs
            if uid in self.response_buffer:
                raise HTTPException(status_code=409, detail=f"Task ID '{uid}' already exists")

            self.response_buffer[uid] = ResponseBufferItem(event=asyncio.Event())

            try:
                self._submit_request(payload, uid)
            except queue.Full:
                self.response_buffer.pop(uid, None)
                raise HTTPException(status_code=503, detail="Server busy, try again later")

            return {"task_id": uid}
        except HTTPException:
            raise

    def _submit_request(self, payload, uid: str) -> None:
        self.request_queue.put_nowait(
            (self.response_queue_id, uid, time.monotonic(), payload)
        )
        self._server._callback_runner.trigger_event(
            EventTypes.ON_REQUEST.value,
            active_requests=self._server.active_requests,
            litserver=self._server,
        )
        logger.debug("Submitted task uid=%s", uid)

    def get_status(self, task_id: str):
        item = self.response_buffer.get(task_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        status = "completed" if item.event.is_set() else "processing"
        return {"status": status}

    def get_result(self, task_id: str):
        item = self.response_buffer.get(task_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        if not item.event.is_set():
            raise HTTPException(status_code=202, detail=f"Task '{task_id}' is still processing")

        response, status = item.response
        if status == LitAPIStatus.ERROR:
            if isinstance(response, HTTPException):
                raise response
            raise HTTPException(status_code=500, detail=str(response))

        # Entry stays in buffer until DELETE is called (by design)
        return response

    def task_done(self, task_id: str):
        item = self.response_buffer.get(task_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        if not item.event.is_set():
            raise HTTPException(
                status_code=409,
                detail=f"Task '{task_id}' is still processing, cannot delete",
            )

        self.response_buffer.pop(task_id, None)
        self._server._callback_runner.trigger_event(
            EventTypes.ON_RESPONSE.value, litserver=self._server
        )
        return {"detail": f"Task '{task_id}' marked as done"}

    async def _prepare_request(self, request, request_type):
        # Unwrap Annotated[T, ...] and Optional[T] before identity check
        rtype = request_type
        origin = get_origin(rtype)
        if origin is Annotated:
            rtype = get_args(rtype)[0]
        origin = get_origin(rtype)
        if origin is Union:
            for arg in get_args(rtype):
                if arg is not type(None):
                    rtype = arg
                    break

        if rtype is Request:
            content_type = request.headers.get("Content-Type", "")
            if content_type == "application/x-www-form-urlencoded" or content_type.startswith(
                "multipart/form-data"
            ):
                return await request.form()
            return await request.json()

        return request
