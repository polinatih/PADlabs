"""
Тест индексации без скачивания моделей: используется заглушка hash-test.
Запуск:  pixi run -e dev pytest
"""

import json
import sys
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "embeddings"))

from build_index import build_one, collection_name  # noqa: E402
from embedders import Embedder  # noqa: E402

CHUNKS = [
    {"chunk_id": "iron_man__headers__000", "article_title": "Iron Man", "article_url": "u1",
     "tag": "film", "strategy": "headers", "section": "Plot", "chunk_index": 0,
     "text": "Tony Stark builds a powered suit of armor and becomes Iron Man"},
    {"chunk_id": "thor__headers__000", "article_title": "Thor", "article_url": "u2",
     "tag": "film", "strategy": "headers", "section": "Plot", "chunk_index": 0,
     "text": "Thor is banished from Asgard to Earth and wields the hammer Mjolnir"},
    {"chunk_id": "loki__headers__000", "article_title": "Loki", "article_url": "u3",
     "tag": "tv_series", "strategy": "headers", "chunk_index": 0,  # без section
     "text": "Loki escapes custody and joins the Time Variance Authority"},
]


def _prepare(tmp_path):
    (tmp_path / "processed").mkdir()
    with open(tmp_path / "processed" / "headers.jsonl", "w", encoding="utf-8") as f:
        for c in CHUNKS:
            f.write(json.dumps(c) + "\n")
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"))


def test_index_and_search(tmp_path):
    client = _prepare(tmp_path)
    build_one(client, "headers", "hash-test", tmp_path, None, 2, {})
    col = client.get_collection(collection_name("headers", "hash-test"))
    assert col.count() == 3

    q = Embedder("hash-test").embed_query("who wields the hammer Mjolnir")
    res = col.query(query_embeddings=[q.tolist()], n_results=1)
    assert res["ids"][0][0] == "thor__headers__000"


def test_rerun_does_not_duplicate(tmp_path):
    client = _prepare(tmp_path)
    cache = {}
    build_one(client, "headers", "hash-test", tmp_path, None, 2, cache)
    build_one(client, "headers", "hash-test", tmp_path, None, 2, cache)
    assert client.get_collection("headers__hash-test").count() == 3


def test_metadata_filter(tmp_path):
    client = _prepare(tmp_path)
    build_one(client, "headers", "hash-test", tmp_path, None, 2, {})
    col = client.get_collection("headers__hash-test")
    q = Embedder("hash-test").embed_query("Time Variance Authority")
    res = col.query(query_embeddings=[q.tolist()], n_results=3, where={"tag": "tv_series"})
    assert res["ids"][0] == ["loki__headers__000"]
