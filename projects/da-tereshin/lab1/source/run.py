import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import requests
import yaml


def main():

    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload = {
        **(config.get("parameters") or {}),
        "model": config["model"],
        "messages": [{"role": "user", "content": config["prompt"]}],
        "stream": False,
    }

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "request": payload,
    }
    output = config_path.parent / config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with output.open("a", encoding="utf-8") as log:
        started = perf_counter()
        try:
            response = requests.post(
                config["base_url"].rstrip("/") + "/v1/chat/completions",
                json=payload, timeout=config["timeout"],
            )
            record["http_status"] = response.status_code
            record["response"] = response.text
            response.raise_for_status()
            record["response"] = response.json()
        except (requests.RequestException, ValueError) as error:
            record["error"] = str(error)
            raise
        finally:
            record["elapsed_seconds"] = perf_counter() - started
            log.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n\n")
    print(f"Результат сохранён: {output.resolve()}")


if __name__ == "__main__":
    main()
