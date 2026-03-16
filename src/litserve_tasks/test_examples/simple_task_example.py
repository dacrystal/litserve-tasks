"""Simple example server demonstrating TaskSpec.

Run standalone:
    python src/litserve_tasks/test_examples/simple_task_example.py

Then test with:
    curl -X POST http://localhost:8000/tasks -H "Content-Type: application/json" -d '{"input": 4}'
    curl http://localhost:8000/tasks/<task_id>
    curl http://localhost:8000/tasks/<task_id>/result
    curl -X DELETE http://localhost:8000/tasks/<task_id>
"""
from typing import Annotated

import litserve as ls
from fastapi import Form
from pydantic import BaseModel

from litserve_tasks import TaskIDField, TaskSpec


class SimpleRequest(BaseModel):
    task_id: TaskIDField
    input: int


class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: SimpleRequest):
        return request.input

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output):
        return {"output": output}


class SimpleFormRequest(BaseModel):
    task_id: TaskIDField
    input: int
    description: str = "default"


class SimpleFormLitAPI(ls.LitAPI):

    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: Annotated[SimpleFormRequest, Form()]):
        return request.input

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output):
        return {"output": output}


if __name__ == "__main__":
    api = SimpleLitAPI(spec=TaskSpec())
    server = ls.LitServer(api, accelerator="cpu")
    server.run(port=8000)
