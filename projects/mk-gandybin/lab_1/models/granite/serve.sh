#!/usr/bin/env bash
exec vllm serve lmstudio-community/granite-4.2-8b-MLX-4bit --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.5 --max-model-len 15000
