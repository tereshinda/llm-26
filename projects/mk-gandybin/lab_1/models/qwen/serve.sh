#!/usr/bin/env bash
exec vllm serve mlx-community/Qwen3.5-4B-OptiQ-4bit --language-model-only --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.5 --max-model-len 24000
