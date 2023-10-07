from .base import Base

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Delta(Base):
    content: str


@dataclass
class Choice(Base):
    index: int
    delta: Delta
    finished_reason: str | None


@dataclass
class OpenAIResponse(Base):
    id: str
    object: str
    created: str
    model: str
    choices: list[Choice]

    @classmethod
    def parse(_, data: dict) -> "OpenAIResponse":
        def x(y: dict):
            return [(Choice.parse(choice)) for choice in y]

        return super().parse(data, choices=x)


@dataclass
class OpenAIError(Base):
    message: str
    type: str
    param: None | Any
    code: Any


@dataclass
class OpenAIErrorResponse(Base):
    error: OpenAIError

    @classmethod
    def parse(_, data: dict) -> "OpenAIErrorResponse":
        return super().parse(data, error=OpenAIError.parse)
