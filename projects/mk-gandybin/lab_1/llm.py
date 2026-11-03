"""Streaming-клиент локального OpenAI-совместимого сервера."""

from time import perf_counter

from openai import OpenAI
from pydantic import BaseModel, ValidationError


class LLMClient:
    def __init__(self, model: str, base_url: str):
        self.model = model

        self.client = OpenAI(
            base_url=base_url,
            api_key="local",
            max_retries=0,
        )

    def generate(self, prompt: str, **parameters) -> tuple[str, dict]:
        """Вернуть текст и метрики; не задавать отсутствующие параметры генерации."""
        request_start = perf_counter()
        with self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
            **parameters,
        ) as stream:
            return self._collect_response(stream, request_start)

    def generate_json(
        self, prompt: str, schema: type[BaseModel], **parameters
    ) -> tuple[dict, dict]:
        """Запросить JSON по Pydantic-схеме и проверить полученный ответ."""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": schema.model_json_schema(),
            },
        }
        request_start = perf_counter()
        with self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_format,
            stream=True,
            stream_options={"include_usage": True},
            **parameters,
        ) as stream:
            response, metrics = self._collect_response(stream, request_start)

        try:
            parsed = schema.model_validate_json(response)
        except ValidationError as exc:
            raise ValueError(
                f"Ответ модели не соответствует JSON-схеме {schema.__name__}: {exc}"
            ) from exc
        return parsed.model_dump(by_alias=True), metrics

    def _collect_response(self, stream, request_start: float) -> tuple[str, dict]:
        """Собрать streaming-текст и метрики для обоих форматов ответа."""
        parts = []
        first_token_time = None
        usage = None
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    if first_token_time is None:
                        first_token_time = perf_counter()
                    parts.append(content)
        request_end = perf_counter()

        completion_tokens = usage.completion_tokens if usage is not None else None
        decode_time = (
            request_end - first_token_time if first_token_time is not None else None
        )
        metrics = {
            "prompt_tokens": usage.prompt_tokens if usage is not None else None,
            "completion_tokens": completion_tokens,
            "total_tokens": usage.total_tokens if usage is not None else None,
            "ttft_seconds": (
                first_token_time - request_start
                if first_token_time is not None
                else None
            ),
            "total_latency_seconds": request_end - request_start,
            "decode_time_seconds": decode_time,
            "decode_tokens_per_second": (
                completion_tokens / decode_time
                if completion_tokens is not None
                and decode_time is not None
                and decode_time > 0
                else None
            ),
        }
        return "".join(parts), metrics
