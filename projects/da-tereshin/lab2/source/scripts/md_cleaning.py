"""Минимальная нормализация тела Markdown с сохранением YAML-шапки.

Удаляем мягкий перенос, заменяем неразрывные пробелы, очищаем концы
строк и оставляем один завершающий перевод строки. Сохраняем отступы,
абзацы, явные Markdown-переносы и содержимое блоков кода.
"""

import argparse
import re
from pathlib import Path

import yaml


def normalize(text: str) -> str:
    lines = []
    fence = None
    for line in text.splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence is not None:
            lines.append(line)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
            continue
        if marker:
            fence = marker[1]
            lines.append(line)
            continue
        # Четыре пробела или табуляция могут обозначать блок кода.
        if line.startswith(("    ", "\t")):
            lines.append(line)
            continue
        line = line.replace("\u00ad", "").replace("\u00a0", " ").replace("\u202f", " ")
        hard_break = line.endswith("  ") and bool(line.strip())
        lines.append(line.rstrip(" \t") + ("  " if hard_break else ""))
    return "\n".join(lines).rstrip("\n") + "\n"


def clean_corpus(config_path: Path) -> None:
    root = config_path.resolve().parent
    options = yaml.safe_load(config_path.read_text(encoding="utf-8"))["cleaning"]
    source = (root / options["input_dir"]).resolve()
    output = (root / options["output_dir"]).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Входная и выходная папки должны быть раздельными")

    pages = sorted(source.rglob("*.md"))
    changed = 0
    for page in pages:
        original = page.read_text(encoding="utf-8")
        _, header, body = original.split("---\n", 2)
        cleaned = "---\n" + header + "---\n" + normalize(body)
        target = output / page.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(cleaned, encoding="utf-8")
        changed += cleaned != original
    print(f"Страниц: {len(pages)}, изменено: {changed} → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Минимальная очистка Markdown")
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[2] / "config.yaml",
    )
    clean_corpus(parser.parse_args().config)