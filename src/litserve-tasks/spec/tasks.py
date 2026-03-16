import asyncio
from functools import cache
import inspect
import time
from typing import Annotated, Generic, TypeVar, Union, get_args, get_origin
import uuid
import logging

from fastapi import Depends, Form, HTTPException, Header, Request, Response, UploadFile
from fastapi import status as status_code

import litserve as ls
from litserve.specs.base import LitSpec
from litserve.callbacks.base import EventTypes
from litserve.utils import LitAPIStatus, ResponseBufferItem
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

TaskIDField = Annotated[str, Field(default_factory=str(uuid.uuid4()), 
                                   json_schema_extra={"TASK_ID": True})]
                                   
class AsyncTaskSpec(LitSpec):
    def __init__(self):
        super().__init__()
        self.api_path = "/tasks"

    @cache  
    def get_task_id_field(self):
        request_type = self._request_type

        # Resolve Annotated[T, ...]
        origin = get_origin(request_type)
        if origin is Annotated:
            request_type = get_args(request_type)[0]

        # Resolve Optional[T] or Union[T, None]
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
    
    def get_task_id(self, request):
        task_id_field = self.get_task_id_field()
        if task_id_field is not None:
            return getattr(request, task_id_field)
        return str(uuid.uuid4())

    def pre_setup(self, lit_api):
        decode_request_signature = inspect.signature(lit_api.decode_request)
        encode_response_signature = inspect.signature(lit_api.encode_response)

        request_type = decode_request_signature.parameters["request"].annotation
        if request_type == decode_request_signature.empty:
            request_type = Request

        response_type = encode_response_signature.return_annotation
        if response_type == encode_response_signature.empty:
            response_type = Response


        self.submit_task.__func__.__annotations__ = self.submit_task.__func__.__annotations__ | {
            "request": request_type,
            "return": response_type
        }
        
        self._request_type = request_type
        self._response_type = response_type

        self.get_task_id_field()

        self.add_endpoint("/tasks", self.submit_task, ["POST"])
        self.add_endpoint("/tasks/{task_id}", self.get_status, ["GET"])
        self.add_endpoint("/tasks/{task_id}", self.task_done, ["DELETE"])
        self.add_endpoint("/tasks/{task_id}/result", self.get_result, ["GET"])

    def decode_request(self, request, **kwargs):
        return request
    
    def encode_response(self, output, **kwargs):
         return output
    
    async def submit_task(self, request ):
        return await self._submit_task(request, self._request_type)

    async def _submit_task(self, request, request_type):
        try:
            logger.debug(f"Handling request: {request}")

            # Prepare request
            payload = await self._prepare_request(request, request_type)

            uid = self.get_task_id(payload)

            # Submit to worker
            await self._submit_request(payload, uid=uid)

            # Wait for response
            event = asyncio.Event()
            self.response_buffer[uid] = ResponseBufferItem(event)

            return {"task_id": uid}

        except HTTPException as e:
            raise e from None
    
    def get_status(self, task_id):
        try:
            uid = task_id
            response_buffer_item = self.response_buffer.get(uid)
            if response_buffer_item is None:
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
            
            event = response_buffer_item.event
            if event.is_set():
                return {"status": "completed"}
            else:
                return {"status": "processing"}

        except HTTPException as e:
            raise e from None
    
    def get_result(self, task_id):
        try:
            uid = task_id
            response_buffer_item = self.response_buffer.get(uid)
            if response_buffer_item is None:
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
            
            event = response_buffer_item.event
            if not event.is_set():
                raise HTTPException(status_code=202, detail=f"Task {task_id} is still processing")
            
            # Process response
            response, status = response_buffer_item.response

            if status == LitAPIStatus.ERROR and isinstance(response, HTTPException):
                logger.error("Error in request: %s", response.detail)
                raise response

            if status == LitAPIStatus.ERROR:
                logger.error("Error in request: %s", response)
                raise HTTPException(status_code=status_code.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Trigger callback
            # self._server._callback_runner.trigger_event(EventTypes.ON_RESPONSE.value, litserver=self._server)

            return response




        except HTTPException as e:
            raise e from None
        
    
    def task_done(self, task_id):
        try:
            uid = task_id
            response_buffer_item = self.response_buffer.get(uid)
            if response_buffer_item is None:
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
            
            event = response_buffer_item.event
            if not event.is_set():
                raise HTTPException(status_code=202, detail=f"Task {task_id} is still processing")
            

            response_buffer_item = self.response_buffer.pop(uid)

            # Trigger callback
            self._server._callback_runner.trigger_event(EventTypes.ON_RESPONSE.value, litserver=self._server)

            return {"detail": f"Task {task_id} marked as done"}

        except HTTPException as e:
            raise e from None

    async def _prepare_request(self, request, request_type) -> dict:
        """Common request preparation logic."""
        if request_type == Request:
            content_type = request.headers.get("Content-Type", "")
            if content_type == "application/x-www-form-urlencoded" or content_type.startswith("multipart/form-data"):
                return await request.form()
            return await request.json()
        return request

    async def _submit_request(self, payload: dict, uid = str(uuid.uuid4())) -> tuple[str, asyncio.Event]:
        """Submit request to worker queue."""
        request_queue = self.request_queue
        response_queue_id = self.response_queue_id

        # Trigger callback
        self._server._callback_runner.trigger_event(
            EventTypes.ON_REQUEST.value,
            active_requests=self._server.active_requests,
            litserver=self._server,
        )

        request_queue.put((response_queue_id, uid, time.monotonic(), payload))
        logger.debug(f"Submitted request uid={uid}")
        return uid, response_queue_id
    
