"""Сохранение результатов эксперимента в CSV и JSON."""

import csv
import json
from pathlib import Path


def save_results(
    records: list[dict], output_dir: Path, model: str
) -> tuple[Path, Path]:
    """Сохранить записи одной модели, сохранив порядок полей первой записи."""
    csv_path = output_dir / f"{model}.csv"
    json_path = output_dir / f"{model}.json"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            for record in records:
                row = record.copy()
                for field in ("response", "expected"):
                    if isinstance(row[field], (dict, list)):
                        row[field] = json.dumps(row[field], ensure_ascii=False)
                writer.writerow(row)

        with json_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)
            file.write("\n")
    except OSError as exc:
        raise OSError(
            f"Не удалось сохранить результаты модели {model} в {output_dir}: {exc}"
        ) from exc

    return csv_path, json_path
