"""Прогон трёх задач для одной уже запущенной локальной модели."""

import argparse
from pathlib import Path

import yaml
from llm import LLMClient
from openai import APIConnectionError, APIStatusError
from prompts import PROMPTS
from result import save_results


def run_experiment(model: str, config: dict) -> list[dict]:
    model_config = config["models"][model]
    client = LLMClient(model_config["name"], model_config["base_url"])
    repetitions = config["experiment"]["repetitions"]
    records = []

    for prompt_name, task in PROMPTS.items():
        for mode, parameters in config["generation"].items():
            request_parameters = dict(parameters)
            if "chat_template_kwargs" in model_config:
                request_parameters["extra_body"] = {
                    "chat_template_kwargs": model_config["chat_template_kwargs"]
                }
            for repetition in range(1, repetitions + 1):
                print(f"{model}: {prompt_name}, {mode}, {repetition}/{repetitions}")
                if task["type"] == "json":
                    response, metrics = client.generate_json(
                        task["prompt"], task["schema"], **request_parameters
                    )
                    expected = task["expected"]
                    score = sum(
                        key in response and response[key] == value
                        for key, value in expected.items()
                    ) / len(expected)
                else:
                    response, metrics = client.generate(
                        task["prompt"], **request_parameters
                    )
                    score = None

                records.append(
                    {
                        "model": model_config["name"],
                        "prompt_name": prompt_name,
                        "prompt_type": task["type"],
                        "mode": mode,
                        "repetition": repetition,
                        "temperature": parameters.get("temperature"),
                        "top_p": parameters.get("top_p"),
                        "max_tokens": parameters.get("max_tokens"),
                        **metrics,
                        "response": response,
                        "expected": task["expected"],
                        "score": score,
                    }
                )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнение локальных LLM: 3 задачи × режимы из config.yaml."
    )
    parser.add_argument("--model", required=True, choices=("gemma", "qwen", "granite"))
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parent

    try:
        with (project_dir / "config.yaml").open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except (OSError, yaml.YAMLError) as exc:
        parser.exit(1, f"Не удалось прочитать config.yaml: {exc}\n")

    if args.model not in config["models"]:
        parser.error(f"Модель {args.model!r} отсутствует в config.yaml.")
    repetitions = config["experiment"]["repetitions"]
    if not isinstance(repetitions, int) or repetitions < 1:
        parser.error("experiment.repetitions должно быть положительным целым числом.")

    try:
        records = run_experiment(args.model, config)
        csv_path, json_path = save_results(records, project_dir / "results", args.model)
    except APIConnectionError as exc:
        base_url = config["models"][args.model]["base_url"]
        parser.exit(1, f"Inference server недоступен ({base_url}): {exc}\n")
    except APIStatusError as exc:
        parser.exit(1, f"Ошибка API inference server (HTTP {exc.status_code}): {exc}\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")

    print(f"Сохранено {len(records)} запусков: {csv_path} и {json_path}")


if __name__ == "__main__":
    main()
