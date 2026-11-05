import argparse
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
import yaml


def check_index(config_path: Path) -> None:
    root = config_path.resolve().parent
    options = yaml.safe_load(config_path.read_text(encoding="utf-8"))["vector_store"]
    target = root / options["output_dir"]
    source = root / options["embeddings_dir"]
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    embedding_manifest = json.loads(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    raw = (target / "chunks.jsonl").read_bytes()
    chunks = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    vectors = np.ascontiguousarray(
        np.load(source / "vectors.npy", allow_pickle=False), dtype=np.float32
    )
    index = faiss.read_index(str(target / "index.faiss"))

    # Порядок чанков определяет смысл числовых меток FAISS.
    assert raw == (source / "chunks.jsonl").read_bytes(), "Копия чанков отличается"
    assert hashlib.sha256(raw).hexdigest() == manifest["chunks_sha256"]
    assert manifest["chunks_sha256"] == embedding_manifest["chunks_sha256"]
    assert hashlib.sha256((source / "vectors.npy").read_bytes()).hexdigest() == manifest["vectors_sha256"]
    assert manifest["embedding_manifest"] == embedding_manifest
    assert list(vectors.shape) == embedding_manifest["shape"]
    assert index.ntotal == manifest["count"] == len(chunks) == len(vectors) > 0
    assert index.d == manifest["dimension"] == vectors.shape[1]
    assert index.is_trained
    assert index.metric_type == faiss.METRIC_L2
    assert (target / "index.faiss").stat().st_size == manifest["index_bytes"]
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
    print(f"Артефакты согласованы: {index.ntotal} векторов × {index.d}")

    # Проверяем поиск по всем векторам корпуса без повторного запуска модели.
    faiss.omp_set_num_threads(1)
    exact = faiss.IndexFlatL2(index.d)
    exact.add(vectors)
    k = min(5, index.ntotal)
    distances, labels = index.search(vectors, k)
    _, expected = exact.search(vectors, k)
    assert np.isfinite(distances).all()
    assert ((labels >= 0) & (labels < len(chunks))).all()
    assert all(len(set(row)) == k for row in labels)
    overlap = np.mean([
        len(set(actual) & set(reference)) / k
        for actual, reference in zip(labels, expected)
    ])
    print(f"Среднее пересечение top-{k} с точным поиском: {overlap:.2%}")
    if manifest["parameters"]["type"] == "flat":
        # При равных расстояниях допустим другой порядок меток.
        exact_distances, _ = exact.search(vectors, k)
        assert np.allclose(distances, exact_distances, atol=1e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка сохранённого FAISS-индекса")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "config.yaml")
    check_index(parser.parse_args().config)