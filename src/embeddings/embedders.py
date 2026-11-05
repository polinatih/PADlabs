"""
Обёртка над embedding-моделями.

Зачем отдельный файл: одну и ту же обёртку используют и скрипт построения
индекса (build_index.py), и потом retriever (чтобы вопрос пользователя
превращался в вектор ТОЙ ЖЕ моделью, что и документы - иначе поиск не работает).

Модели (обе бесплатные, локальные, работают на CPU, статьи у нас на английском):

  minilm : sentence-transformers/all-MiniLM-L6-v2
           384 измерения, макс. вход 256 токенов. Очень быстрая, лёгкая.
  bge    : BAAI/bge-small-en-v1.5
           384 измерения, макс. вход 512 токенов. Чуть медленнее, в бенчмарках
           поиска (MTEB) обычно заметно сильнее MiniLM. Для запросов рекомендует
           специальный префикс - он добавлен ниже в query_prefix.

hash-test - НЕ настоящая модель, а детерминированная заглушка (хэширование слов).
Нужна только для тестов без скачивания моделей. В экспериментах не использовать.
"""

import hashlib
import re

import numpy as np

MODELS = {
    "minilm": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "max_seq": 256,
        "doc_prefix": "",
        "query_prefix": "",
    },
    "bge": {
        "hf_name": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        "max_seq": 512,
        "doc_prefix": "",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "hash-test": {
        "hf_name": "hash-test (stub)",
        "dim": 384,
        "max_seq": 10_000,
        "doc_prefix": "",
        "query_prefix": "",
    },
}


class Embedder:
    def __init__(self, key: str):
        if key not in MODELS:
            raise ValueError(f"Неизвестная модель '{key}'. Доступны: {list(MODELS)}")
        self.key = key
        self.cfg = MODELS[key]
        self.dim = self.cfg["dim"]
        self._model = None  # sentence-transformers грузим лениво

    # ---------- загрузка ----------
    def _load(self):
        if self._model is None:
            # импорт здесь, чтобы тесты с hash-test не требовали torch
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.cfg["hf_name"])
            self._model.max_seq_length = self.cfg["max_seq"]
        return self._model

    # ---------- эмбеддинги ----------
    def _encode(self, texts, batch_size):
        if self.key == "hash-test":
            return np.vstack([_hash_vector(t, self.dim) for t in texts])
        model = self._load()
        return model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # единичные векторы -> cosine = скалярное произведение
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def embed_documents(self, texts, batch_size: int = 64) -> np.ndarray:
        p = self.cfg["doc_prefix"]
        return self._encode([p + t for t in texts], batch_size)

    def embed_query(self, text: str) -> np.ndarray:
        p = self.cfg["query_prefix"]
        return self._encode([p + text], 1)[0]

    # ---------- диагностика ----------
    def count_truncated(self, texts) -> int:
        """
        Сколько текстов длиннее лимита модели (они будут молча обрезаны).
        Важно для исследования: у MiniLM лимит 256 токенов, а chunks по
        заголовкам в среднем ~1365 символов (~330 токенов) -> часть текста
        MiniLM просто не увидит.
        """
        if self.key == "hash-test":
            return 0
        model = self._load()
        ids = model.tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
        return sum(1 for x in ids if len(x) > self.cfg["max_seq"])


def _hash_vector(text: str, dim: int) -> np.ndarray:
    """Заглушка: мешок слов, слова хэшируются в индексы, вектор нормируется."""
    v = np.zeros(dim, dtype=np.float32)
    for w in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
        v[h % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v
