import argparse
import hashlib
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml
from huggingface_hub import model_info
from sentence_transformers import SentenceTransformer


def embed(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    options = config["embedding"]
    if options["type"] != "dense":
        raise ValueError("Пока реализован только embedding.type=dense")
    if type(options["batch_size"]) is not int or options["batch_size"] <= 0:
        raise ValueError("batch_size должен быть положительным целым числом")
    root = config_path.resolve().parent
    chunks_dir = root / options["chunks_dir"]
    raw = (chunks_dir / "chunks.jsonl").read_bytes()
    chunks = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not chunks or any(not chunk["text"].strip() for chunk in chunks):
        raise ValueError("Нужен непустой список непустых чанков")
    if len({chunk["id"] for chunk in chunks}) != len(chunks):
        raise ValueError("Повторяющиеся id чанков")
    texts = [chunk["text"] for chunk in chunks]

    revision = model_info(options["model"], revision=options["revision"]).sha
    print(f"Загрузка {options['model']} ({revision}), устройство {options['device']}", flush=True)
    model = SentenceTransformer(
        options["model"], revision=revision, device=options["device"]
    )
    # Проверяем реальные токены модели, включая специальные токены.
    lengths = [len(model.tokenizer(text, truncation=False)["input_ids"]) for text in texts]
    limit = model.max_seq_length
    too_long = [chunks[i]["id"] for i, length in enumerate(lengths) if length > limit]
    if too_long:
        raise ValueError(f"Лимит модели {limit} токенов превышен: {too_long[:5]}")
    print(f"Чанков: {len(chunks)}, максимум токенов модели: {max(lengths)}/{limit}")
    vectors = model.encode(
        texts,
        batch_size=options["batch_size"],
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        precision="float32",
        prompt="",
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError(f"Неожиданная форма векторов: {vectors.shape}")
    if not np.isfinite(vectors).all() or not np.allclose(
        np.linalg.norm(vectors, axis=1), 1, atol=1e-4
    ):
        raise ValueError("Векторы содержат некорректные значения или не нормализованы")

    created_at = datetime.now(ZoneInfo("Europe/Moscow"))
    run_id = created_at.strftime("%Y-%m-%d_%H-%M-%S_%f")
    target = root / options["output_dir"] / run_id
    target.mkdir(parents=True, exist_ok=False)
    np.save(target / "vectors.npy", vectors, allow_pickle=False)
    (target / "chunks.jsonl").write_bytes(raw)
    manifest = {
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "embedding": options,
        "model_revision": revision,
        "normalized": True,
        "dtype": str(vectors.dtype),
        "shape": list(vectors.shape),
        "max_model_tokens": max(lengths),
        "model_token_limit": limit,
        "chunks_sha256": hashlib.sha256(raw).hexdigest(),
        "chunks_manifest": json.loads((chunks_dir / "manifest.json").read_text(encoding="utf-8")),
        "packages": {name: version(name) for name in (
            "sentence-transformers", "transformers", "torch", "numpy", "huggingface-hub"
        )},
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Сохранено {vectors.shape[0]} векторов размерности {vectors.shape[1]} → {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Чанки → dense-эмбеддинги")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "config.yaml")
    embed(parser.parse_args().config)