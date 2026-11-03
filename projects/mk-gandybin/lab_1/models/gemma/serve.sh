#!/usr/bin/env bash
exec vllm serve mlx-community/gemma-4-E4B-it-qat-4bit --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.5 --max-model-len 24000
