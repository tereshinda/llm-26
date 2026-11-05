import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import faiss
import numpy as np
import yaml


def load_to_vector_store(
    config_path: Path, *, drop_index: bool = False, rebuild: bool = False
) -> Path:
    if drop_index and rebuild:
        raise ValueError("Выбери только один флаг: --drop-index или --rebuild")
    root = config_path.resolve().parent
    options = yaml.safe_load(config_path.read_text(encoding="utf-8"))["vector_store"]
    target = root / options["output_dir"]
    names = ("index.faiss", "chunks.jsonl", "manifest.json")
    if drop_index or rebuild:
        # Удаляем только файлы индекса и их временные версии.
        for name in names:
            (target / name).unlink(missing_ok=True)
            (target / f"{name}.tmp").unlink(missing_ok=True)
        print(f"Индекс удалён (если существовал): {target}")
    if drop_index:
        return target

    source = root / options["embeddings_dir"]
    embedding_manifest = json.loads((source / "manifest.json").read_text())
    raw = (source / "chunks.jsonl").read_bytes()
    chunks = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    vector_raw = (source / "vectors.npy").read_bytes()
    vectors = np.ascontiguousarray(
        np.load(source / "vectors.npy", allow_pickle=False), dtype=np.float32
    )
    # Проверяем соответствие векторов и метаданных до построения индекса.
    assert list(vectors.shape) == embedding_manifest["shape"]
    assert vectors.shape[0] == len(chunks) > 0
    assert hashlib.sha256(raw).hexdigest() == embedding_manifest["chunks_sha256"]
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)

    kind = options["type"]
    parameters = {"type": kind, "metric": "squared_l2"}
    if kind in {"hnsw", "hnsw_pq"}:
        parameters.update({key: options[key] for key in
                           ("M", "ef_construction", "ef_search")})
    if kind == "hnsw_pq":
        parameters.update({key: options[key] for key in ("pq_m", "pq_nbits")})

    identity = {
        "parameters": parameters,
        "vectors_sha256": hashlib.sha256(vector_raw).hexdigest(),
        "chunks_sha256": hashlib.sha256(raw).hexdigest(),
        "faiss_version": faiss.__version__,
        "embedding_manifest": embedding_manifest,
    }
    if any((target / name).exists() for name in names):
        if not all((target / name).exists() for name in names):
            raise ValueError("Индекс сохранён не полностью; запусти с --rebuild")
        saved = json.loads((target / "manifest.json").read_text())
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError("Данные или параметры изменились; запусти с --rebuild")
        print(f"Индекс уже существует: {target}")
        return target

    dimension = vectors.shape[1]
    if kind == "flat":
        index = faiss.IndexFlatL2(dimension)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, options["M"])
    elif kind == "hnsw_pq":
        index = faiss.IndexHNSWPQ(
            dimension, options["pq_m"], options["M"], options["pq_nbits"]
        )
    else:
        raise ValueError(f"Неизвестный тип индекса: {kind}")

    # Для воспроизводимости результатов
    faiss.omp_set_num_threads(1)
    if kind != "flat":
        index.hnsw.efConstruction = options["ef_construction"]
        index.hnsw.efSearch = options["ef_search"]
    if not index.is_trained:
        index.train(vectors)
    index.add(vectors)

    target.mkdir(parents=True, exist_ok=True)
    # Сначала полностью записываем временные файлы, затем заменяем результаты.
    faiss.write_index(index, str(target / "index.faiss.tmp"))
    (target / "chunks.jsonl.tmp").write_bytes(raw)
    manifest = {
        **identity,
        "created_at": datetime.now(ZoneInfo("Europe/Moscow")).isoformat(),
        "count": index.ntotal,
        "dimension": dimension,
        "index_bytes": (target / "index.faiss.tmp").stat().st_size,
        "id_mapping": "FAISS label = zero-based row in chunks.jsonl",
    }
    (target / "manifest.json.tmp").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name in names:
        (target / f"{name}.tmp").replace(target / name)
    print(f"{kind}: {index.ntotal} векторов × {dimension} → {target}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Готовые эмбеддинги → индекс FAISS")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "config.yaml")
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument("--drop-index", action="store_true", help="Удалить индекс без построения")
    flags.add_argument("--rebuild", action="store_true", help="Удалить индекс и построить заново")
    args = parser.parse_args()
    load_to_vector_store(args.config, drop_index=args.drop_index, rebuild=args.rebuild)