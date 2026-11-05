"""Тест retriever и метрик на заглушке hash-test. Запуск: pixi run -e dev pytest"""

import json
import sys
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[1]
for sub in ("embeddings", "retrieval", "evaluation"):
    sys.path.insert(0, str(ROOT / "src" / sub))

from build_index import build_one  # noqa: E402
from eval_retrieval import evaluate, load_questions, score_question  # noqa: E402
from retriever import Retriever  # noqa: E402

CHUNKS = [
    {"chunk_id": "a0", "article_title": "Iron Man", "article_url": "u1", "tag": "film", "strategy": "headers",
     "chunk_index": 0, "text": "Tony Stark builds a powered suit of armor and becomes Iron Man"},
    {"chunk_id": "b0", "article_title": "Thor", "article_url": "u2", "tag": "film", "strategy": "headers",
     "chunk_index": 0, "text": "Thor is banished from Asgard to Earth and wields the hammer Mjolnir"},
    {"chunk_id": "c0", "article_title": "Loki", "article_url": "u3", "tag": "tv_series", "strategy": "headers",
     "chunk_index": 0, "text": "Loki escapes custody and joins the Time Variance Authority"},
]
QUESTIONS = [
    {"id": "q1", "type": "factual", "question": "who wields the hammer Mjolnir",
     "relevant_articles": ["Thor"]},
    {"id": "q2", "type": "multi_doc", "question": "Time Variance Authority and powered suit of armor",
     "relevant_articles": ["Loki", "Iron Man"]},
    {"id": "q3", "type": "no_answer", "question": "capital of Australia", "relevant_articles": []},
]


def _retriever(tmp_path):
    (tmp_path / "processed").mkdir()
    with open(tmp_path / "processed" / "headers.jsonl", "w", encoding="utf-8") as f:
        for c in CHUNKS:
            f.write(json.dumps(c) + "\n")
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    build_one(client, "headers", "hash-test", tmp_path, None, 2, {})
    return Retriever("headers", "hash-test", data_dir=tmp_path)


def test_search_and_filter(tmp_path):
    r = _retriever(tmp_path)
    hits = r.search("who wields the hammer Mjolnir", k=2)
    assert hits[0]["article_title"] == "Thor" and hits[0]["rank"] == 1
    assert hits[0]["score"] >= hits[1]["score"]
    only_tv = r.search("hammer", k=3, tag="tv_series")
    assert [h["article_title"] for h in only_tv] == ["Loki"]


def test_metrics_values():
    hits = [{"rank": 1, "article_title": "X"}, {"rank": 2, "article_title": "Loki"}, {"rank": 3, "article_title": "Thor"}]
    s = score_question(hits, {"Loki", "Iron Man"}, k=3)
    assert s == {"hit": 1.0, "recall": 0.5, "rr": 0.5}
    assert score_question(hits, {"Iron Man"}, k=3) == {"hit": 0.0, "recall": 0.0, "rr": 0.0}


def test_evaluate_skips_no_answer(tmp_path):
    r = _retriever(tmp_path)
    summary, per_q = evaluate(r, QUESTIONS, [1, 3])
    assert all(row["n_questions"] == 2 for row in summary)
    assert summary[1]["hit_at_k"] == 1.0          # K=3 охватывает все 3 chunks
    assert summary[0]["no_answer_top1_score"] != ""


def test_real_questions_file_loads():
    qs = load_questions(ROOT / "data" / "eval" / "questions.jsonl")
    assert len(qs) >= 30
    assert all(q["relevant_articles"] for q in qs if q["type"] != "no_answer")
