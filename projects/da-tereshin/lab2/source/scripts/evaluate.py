import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


def retrieval_metrics(results: list[dict], references: list[dict]) -> dict:
    gold = {(ref["source"], ref["page"]) for ref in references}
    found = {(row["source"], row["page"]) for row in results}
    first = next((row["rank"] for row in results
                  if (row["source"], row["page"]) in gold), None)
    return {
        "hit": float(first is not None),
        "mrr": 1 / first if first is not None else 0.0,
        "recall": len(found & gold) / len(gold),
    }


def evaluate(config_path: Path) -> Path:
    root = config_path.resolve().parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    options = config["evaluation"]
    store = root / config["vector_store"]["output_dir"]
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    raw_chunks = (store / "chunks.jsonl").read_bytes()
    chunks = [json.loads(line) for line in raw_chunks.decode("utf-8").splitlines()]
    raw_questions = (root / options["questions"]).read_bytes()
    questions = json.loads(raw_questions)
    index = faiss.read_index(str(store / "index.faiss"))
    assert hashlib.sha256(raw_chunks).hexdigest() == manifest["chunks_sha256"]
    assert index.ntotal == len(chunks) == manifest["count"]
    assert index.d == manifest["dimension"]
    assert options["k"] > 0
    available = {(c["metadata"]["source"], c["metadata"]["page"]) for c in chunks}
    for question in questions:
        assert question["references"], question["id"]
        for ref in question["references"]:
            assert (ref["source"], ref["page"]) in available, (question["id"], ref)

    # Кодируем вопросы той же моделью и ревизией, что использованы для корпуса.
    embedding = manifest["embedding_manifest"]
    runtime = config["embedding"]
    model = SentenceTransformer(
        embedding["embedding"]["model"],
        revision=embedding["model_revision"],
        device=runtime["device"],
    )
    queries = model.encode(
        [q["question"] for q in questions],
        batch_size=runtime["batch_size"],
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        precision="float32",
        prompt="",
    )
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    faiss.omp_set_num_threads(1)
    distances, labels = index.search(queries, options["k"])
    rows = []
    for question, query_distances, query_labels in zip(questions, distances, labels):
        results = []
        for rank, (label, distance) in enumerate(zip(query_labels, query_distances), 1):
            if label == -1:
                continue
            chunk = chunks[int(label)]
            results.append({
                "rank": rank,
                "chunk_id": chunk["id"],
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
                "squared_l2": float(distance),
                "text": chunk["text"],
            })
        metrics = retrieval_metrics(results, question["references"])
        rows.append({**question, "metrics": metrics, "results": results})
        print(f"{question['id']}: Hit={metrics['hit']:.0f}, "
              f"MRR={metrics['mrr']:.3f}, Recall={metrics['recall']:.3f}")

    # Каждому вопросу даём одинаковый вес, независимо от числа эталонных страниц.
    metrics = {f"{name}@{options['k']}": float(np.mean([
        row["metrics"][name] for row in rows
    ])) for name in ("hit", "mrr", "recall")}
    now = datetime.now(ZoneInfo("Europe/Moscow"))
    target = root / options["output_dir"] / now.strftime("%Y-%m-%d_%H-%M-%S_%f")
    target.mkdir(parents=True, exist_ok=False)
    report = {
        "created_at": now.isoformat(),
        "question_count": len(questions),
        "k": options["k"],
        "metrics": metrics,
        "questions_sha256": hashlib.sha256(raw_questions).hexdigest(),
        "index_sha256": hashlib.sha256((store / "index.faiss").read_bytes()).hexdigest(),
        "index_manifest": manifest,
        "runtime": {"device": runtime["device"], "batch_size": runtime["batch_size"]},
        "questions": rows,
    }
    (target / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Результаты: {target / 'results.json'}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка retrieval по PDF-страницам")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "config.yaml")
    evaluate(parser.parse_args().config)