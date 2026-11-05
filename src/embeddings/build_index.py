"""
Построение векторного индекса (embeddings + Vector Store).

Что делает:
  Берёт chunks из data/processed/<strategy>.jsonl, считает embeddings
  выбранной моделью и кладёт в ChromaDB.

Для КАЖДОЙ пары (стратегия chunking x модель embeddings) создаётся своя
коллекция Chroma с именем "<strategy>__<model>", например "headers__bge".
Так эксперименты можно сравнивать между собой на одних и тех же вопросах.

Почему ChromaDB (для README, пункт "обоснуйте выбор vector DB"):
  - ставится одной командой, работает локально без сервера и Docker;
  - хранит данные на диске (повторный запуск не пересчитывает всё заново);
  - умеет фильтры по metadata (нужны для фильтрации в retrieval);
  - для ~25 тыс. chunks этого более чем достаточно; FAISS быстрее на миллионах
    векторов, но не хранит metadata и текст - пришлось бы писать это самим.

Повторный запуск безопасен: уже проиндексированные chunk_id пропускаются.

Запуск (из корня lab1, через Pixi):
  pixi run python src/embeddings/build_index.py
  pixi run python src/embeddings/build_index.py --strategies headers --models bge
  pixi run python src/embeddings/build_index.py --limit 200      # быстрая проба
"""

import argparse
import json
import sys
import time
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedders import MODELS, Embedder  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]  # корень lab1 (src/embeddings/ -> lab1)
STRATEGIES = ["fixed", "fixed_overlap", "paragraph", "headers"]


def collection_name(strategy: str, model_key: str) -> str:
    return f"{strategy}__{model_key}"


def load_chunks(path: Path, limit: int | None = None):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
            if limit and len(chunks) >= limit:
                break
    return chunks


def to_metadata(rec: dict) -> dict:
    """Chroma принимает в metadata только str/int/float/bool (None нельзя)."""
    meta = {
        "article_title": rec.get("article_title", ""),
        "article_url": rec.get("article_url", ""),
        "tag": rec.get("tag", ""),
        "strategy": rec.get("strategy", ""),
        "chunk_index": int(rec.get("chunk_index", 0)),
        "n_chars": len(rec.get("text", "")),
    }
    if rec.get("section"):
        meta["section"] = rec["section"]
    return meta


def build_one(client, strategy, model_key, data_dir, limit, batch_size, embedder_cache):
    src = data_dir / "processed" / f"{strategy}.jsonl"
    if not src.exists():
        print(f"  [пропуск] нет файла {src}")
        return

    chunks = load_chunks(src, limit)
    name = collection_name(strategy, model_key)
    cfg = MODELS[model_key]

    col = client.get_or_create_collection(
        name=name,
        metadata={
            "hnsw:space": "cosine",
            "model": cfg["hf_name"],
            "dim": cfg["dim"],
            "strategy": strategy,
        },
    )

    # какие chunks уже в индексе -> не считаем заново
    done = set()
    ids_all = [c["chunk_id"] for c in chunks]
    for i in range(0, len(ids_all), 500):
        done.update(col.get(ids=ids_all[i : i + 500], include=[])["ids"])
    todo = [c for c in chunks if c["chunk_id"] not in done]
    print(f"  {name}: всего {len(chunks)}, уже в индексе {len(done)}, считаю {len(todo)}")
    if not todo:
        return

    embedder = embedder_cache.setdefault(model_key, Embedder(model_key))
    t0 = time.time()
    truncated = 0

    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        texts = [c["text"] for c in batch]
        truncated += embedder.count_truncated(texts)
        vectors = embedder.embed_documents(texts, batch_size=batch_size)
        col.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=[to_metadata(c) for c in batch],
        )
        if (i // batch_size) % 20 == 0:
            print(f"    {min(i + batch_size, len(todo))}/{len(todo)}")

    dt = time.time() - t0
    print(
        f"  готово {name}: {len(todo)} chunks за {dt:.0f} с; "
        f"обрезано моделью (длиннее {cfg['max_seq']} токенов): {truncated} "
        f"({100 * truncated / len(todo):.1f}%)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    ap.add_argument("--models", nargs="+", default=["minilm", "bge"], choices=list(MODELS))
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--limit", type=int, default=None, help="взять только первые N chunks (для пробы)")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=str(args.data_dir / "chroma"))
    cache = {}
    for model_key in args.models:
        for strategy in args.strategies:
            build_one(client, strategy, model_key, args.data_dir, args.limit, args.batch_size, cache)

    print("\nКоллекции в индексе:")
    for c in client.list_collections():
        print(f"  {c.name}: {client.get_collection(c.name).count()} chunks")


if __name__ == "__main__":
    main()
