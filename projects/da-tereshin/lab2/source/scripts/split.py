import argparse
import hashlib
import json
import re
from pathlib import Path

from datetime import datetime
from zoneinfo import ZoneInfo

import tiktoken
import yaml
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def build_chunks(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    options = config["splitter"]
    kind = options["type"]
    size, overlap = options["chunk_size"], options["overlap"]

    # Проверки
    if kind not in {"recursive", "hybrid"}:
        raise ValueError("splitter.type: ожидается recursive или hybrid")
    if type(size) is not int or type(overlap) is not int:
        raise ValueError("chunk_size и overlap должны быть целыми числами")
    if not 100 <= size <= 1000 or not 0 <= overlap < size:
        raise ValueError("Нужно 100 <= chunk_size <= 1000 и 0 <= overlap < chunk_size")
    if type(options["include_headers"]) is not bool:
        raise ValueError("include_headers должен быть true или false")

    root = config_path.resolve().parent
    input_dir = root / config["input_dir"]
    pages = sorted(input_dir.glob("*/*.md"))
    if not pages:
        raise ValueError(f"Не найдены страницы: {input_dir}")

    tokenizer = tiktoken.get_encoding(options["encoding"])

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, disallowed_special=()))

    recursive = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", " ", ""],
    )
    markdown = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=not options["include_headers"],
    )

    chunks = []
    corpus_hash = hashlib.sha256()
    for page_path in pages:
        relative_path = page_path.relative_to(input_dir).as_posix()
        raw = page_path.read_text(encoding="utf-8")
        # JSON-кодирование
        corpus_hash.update(json.dumps([relative_path, raw], ensure_ascii=False).encode())
        parts = raw.split("---\n", 2)

        metadata = yaml.safe_load(parts[1])
        text = parts[2].strip()
        if not text:
            continue

        if kind == "hybrid":
            sections = [(doc.page_content, doc.metadata)
                        for doc in markdown.split_text(text)]
        else:
            if not options["include_headers"]:
                text = re.sub(r"^ {0,3}#{1,3}[ \t]+.*$", "", text, flags=re.MULTILINE)
            sections = [(text, {})]

        for content, headers in sections:
            for chunk in recursive.split_text(content):
                chunks.append({
                    "id": f"{relative_path}:{len(chunks):06d}",
                    "text": chunk,
                    "token_count": count_tokens(chunk),
                    "metadata": {**metadata, "markdown_path": relative_path,
                                 "headers": headers},
                })


    created_at = datetime.now(ZoneInfo("Europe/Moscow"))
    run_id = created_at.strftime("%Y-%m-%d_%H-%M-%S_%f")
    manifest = {
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "config": config,
        "corpus_sha256": corpus_hash.hexdigest(),
    }
    target = root / config["output_dir"] / run_id
    target.mkdir(parents=True, exist_ok=False)
    
    with (target / "chunks.jsonl").open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    manifest["chunk_count"] = len(chunks)
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Страниц: {len(pages)}, чанков: {len(chunks)} → {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Markdown-страницы → чанки JSONL")
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[2] / "config.yaml",
    )
    build_chunks(parser.parse_args().config)