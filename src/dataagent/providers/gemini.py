"""Gemini provider — Google GenAI SDK provider adapter."""
from __future__ import annotations

import json
import time
from typing import Any

from google import genai
from google.genai import types

from dataagent.providers.base import LLMProvider, LLMResponse, Message, ToolCall


class GeminiProvider(LLMProvider):
    """Wrap Gemini and translate between Gemini function calls and internal ToolCall."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file.")

        self._api_key = api_key
        self._model = model
        self._client = genai.Client(api_key=api_key)

    def chat(self, messages: list[Message], tools: list[dict]) -> LLMResponse:
        started = time.perf_counter()

        system_instruction, contents = self._messages_to_gemini_contents(messages)
        function_declarations = [
            self._tool_schema_to_function_declaration(tool) for tool in tools
        ]

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction or None,
                tools=[
                    types.Tool(
                        function_declarations=function_declarations,
                    )
                ]
                if function_declarations
                else None,
                temperature=0.1,
                max_output_tokens=2048,
            ),
        )

        message = self._response_to_message(response)

        usage = getattr(response, "usage_metadata", None)

        return LLMResponse(
            message=message,
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            latency_ms=(time.perf_counter() - started) * 1000,
            model=self._model,
            provider="gemini",
        )

    def _messages_to_gemini_contents(
        self,
        messages: list[Message],
    ) -> tuple[str, list[types.Content]]:
        system_parts: list[str] = []
        contents: list[types.Content] = []

        for message in messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)
                continue

            if message.role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=message.content or "")],
                    )
                )
                continue

            if message.role == "assistant":
                parts: list[types.Part] = []

                if message.content:
                    parts.append(types.Part.from_text(text=message.content))

                for tool_call in message.tool_calls:
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                name=tool_call.name,
                                args=tool_call.arguments,
                            )
                        )
                    )

                if parts:
                    contents.append(types.Content(role="model", parts=parts))

                continue

            if message.role == "tool":
                response_payload: dict[str, Any]

                try:
                    response_payload = json.loads(message.content or "{}")
                except json.JSONDecodeError:
                    response_payload = {"result": message.content or ""}

                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=message.name or "tool",
                                response=response_payload,
                            )
                        ],
                    )
                )

        return "\n\n".join(system_parts), contents

    def _response_to_message(self, response: Any) -> Message:
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        candidate = response.candidates[0] if response.candidates else None

        if not candidate or not candidate.content or not candidate.content.parts:
            return Message(role="assistant", content="")

        for index, part in enumerate(candidate.content.parts):
            function_call = getattr(part, "function_call", None)

            if function_call:
                tool_calls.append(
                    ToolCall(
                        id=f"gemini_call_{index}",
                        name=function_call.name,
                        arguments=dict(function_call.args or {}),
                    )
                )
                continue

            text = getattr(part, "text", None)

            if text:
                content_parts.append(text)

        return Message(
            role="assistant",
            content="\n".join(content_parts).strip(),
            tool_calls=tool_calls,
        )

    def _tool_schema_to_function_declaration(
        self,
        tool: dict[str, Any],
    ) -> types.FunctionDeclaration:
        parameters = self._clean_schema_for_gemini(tool.get("parameters", {}))

        return types.FunctionDeclaration(
            name=tool["name"],
            description=tool.get("description", ""),
            parameters=parameters,
        )

    def _clean_schema_for_gemini(self, schema: Any) -> Any:
        if isinstance(schema, list):
            return [self._clean_schema_for_gemini(item) for item in schema]

        if not isinstance(schema, dict):
            return schema

        cleaned: dict[str, Any] = {}

        for key, value in schema.items():
            if key == "type" and isinstance(value, list):
                non_null = [item for item in value if item != "null"]
                cleaned[key] = non_null[0] if non_null else "string"
                continue

            if key == "default":
                continue

            cleaned[key] = self._clean_schema_for_gemini(value)

        return cleaned

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "gemini"