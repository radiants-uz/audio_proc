from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LlmUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, other: "LlmUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls


@dataclass
class LlmResponse:
    data: dict
    usage: LlmUsage = field(default_factory=LlmUsage)


class LlmClient(Protocol):
    model_name: str

    async def complete_json(self, system: str, user: str, schema: dict, max_tokens: int) -> LlmResponse: ...
