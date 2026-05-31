"""Groq provider — OpenAI-compatible API with recovery for malformed tool output."""
from __future__ import annotations

import ast
import json
import random
import re
import time
from typing import Any

import groq

from dataagent.providers.base import LLMProvider, LLMResponse, Message, ToolCall


class GroqProvider(LLMProvider):
    """Wrap the Groq SDK and translate provider-specific responses."""

    def __init__(self, api_key: str, model: str, max_tokens: int = 1024) -> None:
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._client = groq.Groq(api_key=api_key, max_retries=0)

    def chat(self, messages: list[Message], tools: list[dict]) -> LLMResponse:
        """Send messages/tools to Groq with bounded retry and tool-call cleanup."""
        msg_dicts = [self._message_to_dict(message) for message in messages]
        start_time = time.perf_counter()

        max_retries = 2
        max_rate_limit_sleep = 5.0
        last_error: Exception | None = None
        response = None

        for attempt in range(max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=msg_dicts,
                    tools=[{"type": "function", "function": tool} for tool in tools]
                    if tools
                    else None,
                    temperature=0.1,
                    max_tokens=self._max_tokens,
                    disable_tool_validation=True,
                )
                break

            except groq.RateLimitError as exc:
                last_error = exc

                if attempt >= max_retries:
                    raise RuntimeError(
                        "Groq rate limit reached. Please wait and try again. "
                        "No tool execution was run for this request."
                    ) from exc

                retry_after = self._retry_after_seconds(exc)

                if retry_after is None:
                    retry_after = (2**attempt) + random.uniform(0.2, 1.0)

                sleep_seconds = min(float(retry_after), max_rate_limit_sleep)

                print(
                    f"[RATE LIMIT] Sleeping {sleep_seconds:.1f}s before retry...",
                    flush=True,
                )

                time.sleep(sleep_seconds)

            except groq.BadRequestError as exc:
                recovered = self._recover_bad_request(
                    error_text=str(exc),
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                )

                if recovered is not None:
                    return recovered

                raise

            except Exception as exc:
                last_error = exc
                raise

        if response is None:
            raise RuntimeError(f"Groq request failed: {last_error}")

        choice = response.choices[0]
        content = choice.message.content or ""

        tool_calls = self._normal_tool_calls(choice.message.tool_calls)

        if not tool_calls:
            tool_calls = self._extract_raw_tool_calls(content)

            if tool_calls:
                content = ""

        message = Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        )

        usage = response.usage

        return LLMResponse(
            message=message,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            model=self._model,
            provider="groq",
        )

    def _normal_tool_calls(self, raw_tool_calls: Any) -> list[ToolCall]:
        """Convert official Groq/OpenAI tool calls into internal ToolCall objects."""
        tool_calls: list[ToolCall] = []

        if not raw_tool_calls:
            return tool_calls

        for tool_call in raw_tool_calls:
            arguments = tool_call.function.arguments

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=arguments or {},
                )
            )

        return tool_calls

    def _recover_bad_request(
        self,
        error_text: str,
        latency_ms: float,
    ) -> LLMResponse | None:
        """Recover when Groq rejects malformed XML-style tool calls."""
        tool_calls = self._extract_raw_tool_calls(error_text)

        if not tool_calls:
            return None

        return LLMResponse(
            message=Message(
                role="assistant",
                content="",
                tool_calls=tool_calls,
            ),
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            model=self._model,
            provider="groq",
        )

    def _extract_raw_tool_calls(self, text: str) -> list[ToolCall]:
        """Extract common malformed Llama/Groq tool-call formats."""
        if not text:
            return []

        parsers = [
            self._extract_function_xml_calls,
            self._extract_python_tag_calls,
        ]

        for parser in parsers:
            tool_calls = parser(text)

            if tool_calls:
                return tool_calls

        return []

    def _extract_function_xml_calls(self, text: str) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []

        pattern = re.compile(
            r"<function=(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)>"
            r"(?P<args>.*?)"
            r"</function>",
            flags=re.DOTALL,
        )

        for index, match in enumerate(pattern.finditer(text)):
            name = match.group("name")
            raw_args = match.group("args")
            arguments = self._parse_arguments(raw_args)

            tool_calls.append(
                ToolCall(
                    id=f"recovered_{name}_{index}",
                    name=name,
                    arguments=arguments,
                )
            )

        return tool_calls

    def _extract_python_tag_calls(self, text: str) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []

        pattern = re.compile(
            r"<\|python_tag\|>\s*"
            r"(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)"
            r"\s*(?P<args>\(.*?\)|\{.*?\})\s*$",
            flags=re.DOTALL,
        )

        match = pattern.search(text.strip())

        if not match:
            return []

        name = match.group("name")
        raw_args = match.group("args").strip()

        if raw_args.startswith("(") and raw_args.endswith(")"):
            raw_args = raw_args[1:-1]

        arguments = self._parse_arguments(raw_args)

        tool_calls.append(
            ToolCall(
                id=f"recovered_{name}_0",
                name=name,
                arguments=arguments,
            )
        )

        return tool_calls

    @staticmethod
    def _parse_arguments(raw_args: str) -> dict[str, Any]:
        raw_args = raw_args.strip()

        if not raw_args:
            return {}

        try:
            parsed = json.loads(raw_args)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

        try:
            parsed = ast.literal_eval(raw_args)

            if isinstance(parsed, dict):
                return parsed

        except (ValueError, SyntaxError):
            pass

        return {}

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        """Read Retry-After from the SDK exception response, when available."""
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)

        if not headers:
            return None

        raw = headers.get("retry-after") or headers.get("Retry-After")

        if raw is None:
            return None

        try:
            return float(raw)
        except ValueError:
            return None

    @staticmethod
    def _message_to_dict(message: Message) -> dict[str, Any]:
        """Convert Message dataclass to Groq SDK message format."""
        if message.role == "assistant":
            payload: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
            }

            if message.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                    for tool_call in message.tool_calls
                ]

            return payload

        if message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "name": message.name,
                "content": message.content or "",
            }

        return {
            "role": message.role,
            "content": message.content or "",
        }

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "groq"