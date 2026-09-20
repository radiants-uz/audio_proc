from typing import Annotated

from pydantic import BaseModel, StringConstraints


class TaskID(BaseModel):
    task_id: str


LangCode = Annotated[
    str,
    StringConstraints(to_lower=True, min_length=2, max_length=8, pattern=r"^[a-z]{2,8}(_[a-z]{2,4})?$"),
]

ServiceName = Annotated[
    str, StringConstraints(min_length=1, max_length=32, pattern=r"^[a-zA-Z0-9\-_.]+$")
]
