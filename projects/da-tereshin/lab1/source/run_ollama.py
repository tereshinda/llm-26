import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import requests
import yaml

config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("config.yaml")
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
options = dict(config.get("parameters") or {})
reasoning = options.pop("reasoning_effort", None)

if reasoning not in (None, "none"):
    raise ValueError("Этот скрипт поддерживает только reasoning_effort: none")
if "max_tokens" in options:
    options["num_predict"] = options.pop("max_tokens")

payload = {
    "model": config["model"],
    "messages": [{"role": "user", "content": config["prompt"]}],
    "options": options,
    "stream": False,
    "keep_alive": 0,
}
if reasoning == "none":
    payload["think"] = False

timestamp = datetime.now(timezone.utc).isoformat()
started = perf_counter()
response = requests.post(config["base_url"].rstrip("/") + "/api/chat",
                         json=payload, timeout=config["timeout"])
elapsed = perf_counter() - started
response.raise_for_status()
answer = response.json()

input_tokens = answer["prompt_eval_count"] - answer.get("prompt_eval_cached_count", 0)
input_seconds = answer["prompt_eval_duration"] / 1e9
output_seconds = answer["eval_duration"] / 1e9
metrics = {
    "prompt_tokens_per_second": input_tokens / input_seconds if input_seconds else None,
    "generation_tokens_per_second": answer["eval_count"] / output_seconds if output_seconds else None,
}
record = {"timestamp": timestamp, "config": config, "request": payload,
          "response": answer, "elapsed_seconds": elapsed, "metrics": metrics}
output = config_path.parent / Path(config["output"]).with_name("native_runs.jsonl")
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("a", encoding="utf-8") as log:
    log.write(json.dumps(record, ensure_ascii=False) + "\n")

print(answer["message"]["content"])
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print(f"Полное время: {elapsed:.2f} с. Результат: {output.resolve()}")