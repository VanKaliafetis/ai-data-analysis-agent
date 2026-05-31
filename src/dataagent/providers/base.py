"""Provider interface contract.

Implement LLMProvider to add a new backend.
The agent loop only ever touches this module's types — never a concrete provider.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """Unified message format shared across all providers."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # populated for role="tool" replies
    name: str | None = None          # tool name, for role="tool" replies


@dataclass
class LLMResponse:
    """Everything the agent needs from one model round-trip."""

    message: Message
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    provider: str


class LLMProvider(ABC):
    """Implement this ABC to add a new LLM backend.

    The agent loop calls only `chat()`. Schema translation, retry logic,
    and auth are each provider's internal concern.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict],
    ) -> LLMResponse:
        """Send messages and tool schemas; return the model's response."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Canonical model identifier string (e.g. 'llama-3.3-70b-versatile')."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short provider slug used in trace logs (e.g. 'groq')."""
        ...
