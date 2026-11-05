import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import copy
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

root = Path(__file__).resolve().parents[2]
scripts = root / "source/scripts"
sys.path.insert(0, str(scripts))
import split
import embed
import evaluate
from load_to_vector_store import load_to_vector_store

revision = "5617a9f61b028005a4858fdac845db406aefb181"
started = time.perf_counter()
model = SentenceTransformer("BAAI/bge-m3", revision=revision, device="cpu",
                            local_files_only=True)
model_seconds = time.perf_counter() - started
embed.model_info = lambda *args, **kwargs: SimpleNamespace(sha=revision)
embed.SentenceTransformer = lambda *args, **kwargs: model
evaluate.SentenceTransformer = lambda *args, **kwargs: model

stamp = datetime.now(ZoneInfo("Europe/Moscow")).strftime("%Y-%m-%d_%H-%M-%S_%f")
suite = root / "experiments" / (stamp + "_chunking")
suite.mkdir(parents=True)
shutil.copytree(scripts, suite / "source/scripts")
shutil.copy2(root / "questions.json", suite / "questions.json")
base = yaml.safe_load((root / "config.yaml").read_text())
rows = []
corpus_hashes = set()
for kind, size, overlap in [("hybrid", 400, 60), ("hybrid", 700, 100),
                            ("recursive", 700, 100)]:
    name = f"{kind}_{size}_{overlap}"
    run = suite / name
    run.mkdir()
    config = copy.deepcopy(base)
    config["input_dir"] = str(root / "data/cleaned")
    config["output_dir"] = str(run / "chunks")
    config["splitter"].update(type=kind, chunk_size=size, overlap=overlap)
    config["embedding"].update(revision=revision, output_dir=str(run / "embeddings"))
    config["vector_store"]["output_dir"] = str(run / "vector_store")
    config["evaluation"].update(questions=str(suite / "questions.json"),
                                output_dir=str(run / "evaluation"), k=10)
    path = run / "config.yaml"
    def save_config():
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    save_config()
    print(f"\nEXPERIMENT {name}", flush=True)
    start = time.perf_counter()
    split.build_chunks(path)
    chunks_dir = next((run / "chunks").iterdir())
    manifest = json.loads((chunks_dir / "manifest.json").read_text())
    corpus_hashes.add(manifest["corpus_sha256"])
    config["embedding"]["chunks_dir"] = str(chunks_dir)
    save_config()
    embed.embed(path)
    embeddings_dir = next((run / "embeddings").iterdir())
    config["vector_store"]["embeddings_dir"] = str(embeddings_dir)
    save_config()
    load_to_vector_store(path)
    build_seconds = time.perf_counter() - start
    evaluation_start = time.perf_counter()
    report_dir = evaluate.evaluate(path)
    evaluation_seconds = time.perf_counter() - evaluation_start
    report = json.loads((report_dir / "results.json").read_text())
    chunks = [json.loads(line) for line in (chunks_dir / "chunks.jsonl").read_text().splitlines()]
    tokens = {c["id"]: c["token_count"] for c in chunks}
    summary = dict(name=name, chunks=len(chunks),
                   mean_chunk_tokens=float(np.mean(list(tokens.values()))),
                   build_seconds=build_seconds, evaluation_seconds=evaluation_seconds,
                   index_bytes=(run / "vector_store/index.faiss").stat().st_size,
                   results=str(report_dir / "results.json"))
    for k in (5, 10):
        per_question = []
        for q in report["questions"]:
            found = q["results"][:k]
            gold = {(r["source"], r["page"]) for r in q["references"]}
            metrics = evaluate.retrieval_metrics(found, q["references"])
            metrics["precision"] = sum((r["source"], r["page"]) in gold for r in found) / k
            metrics["context_tokens"] = sum(tokens[r["chunk_id"]] for r in found)
            per_question.append(dict(id=q["id"], **metrics))
        summary[f"per_question@{k}"] = per_question
        for metric in ("hit", "mrr", "recall", "precision", "context_tokens"):
            summary[f"{metric}@{k}"] = float(np.mean([q[metric] for q in per_question]))
    rows.append(summary)
    (suite / "comparison.json").write_text(json.dumps(dict(
        model_revision=revision, model_load_seconds=model_seconds,
        questions_sha256=report["questions_sha256"],
        corpus_sha256=manifest["corpus_sha256"], experiments=rows),
        ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)

assert len(corpus_hashes) == 1
lines = ["| Вариант | Чанков | k | Hit | MRR | Recall | Precision | Токенов контекста |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
for row in rows:
    for k in (5, 10):
        values = " | ".join(f"{row[f'{metric}@{k}']:.4f}" for metric in
                            ("hit", "mrr", "recall", "precision"))
        lines.append(f"| {row['name']} | {row['chunks']} | {k} | {values} | "
                     f"{row[f'context_tokens@{k}']:.0f} |")
(suite / "comparison.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
print(f"SUITE: {suite}", flush=True)
