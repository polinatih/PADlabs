"""
Retriever: по вопросу находит Top-K самых похожих chunks в ChromaDB.

    r = Retriever("headers", "bge")
    hits = r.search("Who directed Iron Man?", k=5)
    hits = r.search("...", k=5, tag="film")        # фильтр по metadata

Каждый hit: dict(chunk_id, text, score, article_title, article_url, tag, section, rank).
score = cosine similarity (1 - расстояние Chroma), чем больше, тем ближе.
"""

import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "embeddings"))
from embedders import Embedder  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


class Retriever:
    def __init__(self, strategy: str, model_key: str, data_dir: Path = ROOT / "data",
                 embedder: Embedder | None = None, client=None):
        self.strategy, self.model_key = strategy, model_key
        self.client = client or chromadb.PersistentClient(path=str(Path(data_dir) / "chroma"))
        self.collection = self.client.get_collection(f"{strategy}__{model_key}")
        self.embedder = embedder or Embedder(model_key)

    def search(self, query: str, k: int = 5, tag: str | None = None) -> list[dict]:
        vec = self.embedder.embed_query(query)
        res = self.collection.query(
            query_embeddings=[vec.tolist()],
            n_results=k,
            where={"tag": tag} if tag else None,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for i, cid in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i]
            hits.append({
                "rank": i + 1,
                "chunk_id": cid,
                "text": res["documents"][0][i],
                "score": 1.0 - res["distances"][0][i],
                "article_title": meta.get("article_title", ""),
                "article_url": meta.get("article_url", ""),
                "tag": meta.get("tag", ""),
                "section": meta.get("section", ""),
            })
        return hits
