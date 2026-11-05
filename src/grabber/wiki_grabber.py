"""
Grabber статей Wikipedia про Marvel Cinematic Universe (версия 5).

Что исправлено по сравнению с предыдущей версией:
  1. Раньше с каждой страницы-хаба брались только первые 300 ссылок
     (MAX_LINKS_PER_HUB). Wikipedia отдаёт ссылки по алфавиту, поэтому в датасет
     попадали только названия примерно от A до I, а Thor, Spider-Man,
     Black Panther, Avengers: Endgame и т.д. терялись. Теперь лимит большой
     и все страницы результатов обходятся полностью.
  2. Раньше при ошибке запроса (например 429) статья молча считалась "не по теме"
     и пропадала. Теперь такие статьи запоминаются и запрашиваются повторно;
     в конце печатается, сколько не удалось получить совсем.
  3. Категории запрашиваются пачками по 50 статей вместо одной за раз -
     в ~50 раз меньше запросов, поэтому лимит скорости (429) срабатывает реже.
  4. Редиректы обрабатываются: "Chris Evans (actor)" и "Chris Evans" - одна статья,
     файл не дублируется.
  5. Тег статьи (film / tv_series / character / actor / soundtrack / ...) теперь
     определяется по её категориям, а не по тому, с какого хаба пришла ссылка.
  6. Статьи про персонажей чужих киновселенных (Sony/Fox) отсекаются по названию
     и удаляются из data/raw, если остались от прошлых запусков.

Повторный запуск безопасен: неизменённые статьи пропускаются (по хэшу текста).

Запуск (из корня проекта):
    pixi run python src/grabber/wiki_grabber.py
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone

import requests

# ======================= CONFIG =======================

LANG = "en"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
REQUEST_DELAY_SEC = 0.5
MAX_RETRIES = 4
MAX_LINKS_PER_HUB = 5000     # защита от бесконечного цикла, а не реальное ограничение
CATEGORY_BATCH = 50          # статей в одном запросе категорий (максимум API)
RETRY_PASSES = 2             # сколько раз перезапрашивать упавшие статьи

HUBS = [
    {
        "page": "List of Marvel Cinematic Universe films",
        "category_patterns": [r"marvel cinematic universe"],
        "tag": "film",
    },
    {
        "page": "List of Marvel Cinematic Universe television series (Marvel Studios)",
        "category_patterns": [r"marvel cinematic universe"],
        "tag": "tv_series",
    },
    {
        "page": "List of Marvel Cinematic Universe film actors",
        "category_patterns": [r"\b(19|20)\d{2}\s+births\b"],
        "tag": "actor",
    },
    {
        "page": "Characters of the Marvel Cinematic Universe",
        "category_patterns": [r"marvel cinematic universe"],
        "tag": "character",
    },
]

# Названия статей про НЕ-MCU версии героев (Sony/Fox/New Line). Такие статьи
# проходят фильтр по категориям, но к MCU не относятся.
DENY_TITLE_PATTERNS = [
    r"\(film character\)",
    r"film series( character)?\)",
    r"\(New Line franchise character\)",
]

API_URL = f"https://{LANG}.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "LabRAGGrabber/5.0 (student RAG lab project)"}


# ======================= ЗАПРОСЫ К API =======================

def api_get(params: dict):
    """GET к Wikipedia API. При 429 и сетевых ошибках повторяет запрос."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=20)

            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5))
                print(f"  [лимит скорости] жду {wait} сек. (попытка {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.RequestException as e:
            print(f"  [ошибка запроса] {e} (попытка {attempt}/{MAX_RETRIES})")
            time.sleep(2 * attempt)

    print("  [ошибка] нет ответа после всех попыток")
    return None


def get_links_from_page(title: str):
    """Все ссылки на статьи (namespace 0) со страницы, со всеми страницами результатов."""
    links = []
    params = {
        "action": "query",
        "prop": "links",
        "titles": title,
        "plnamespace": "0",
        "pllimit": "500",
        "format": "json",
    }
    while True:
        data = api_get(params)
        if not data:
            print(f"  [предупреждение] список ссылок получен не полностью ({len(links)} шт.)")
            break

        for page in data.get("query", {}).get("pages", {}).values():
            links.extend(link["title"] for link in page.get("links", []))

        if len(links) >= MAX_LINKS_PER_HUB:
            links = links[:MAX_LINKS_PER_HUB]
            break

        if "continue" in data:
            params.update(data["continue"])
            time.sleep(REQUEST_DELAY_SEC)
        else:
            break
    return links


def get_categories_batch(titles):
    """
    Категории сразу для нескольких статей одним запросом (до 50).
    Возвращает dict: исходное название -> (итоговое название после редиректа, [категории]),
    либо None, если запрос не удался (тогда статьи надо запросить повторно).
    """
    params = {
        "action": "query",
        "prop": "categories",
        "titles": "|".join(titles),
        "cllimit": "max",
        "redirects": "1",
        "format": "json",
    }
    cats_by_title = {}
    redirect_map = {}
    norm_map = {}

    while True:
        data = api_get(params)
        if data is None:
            return None
        q = data.get("query", {})
        redirect_map.update({r["from"]: r["to"] for r in q.get("redirects", [])})
        norm_map.update({n["from"]: n["to"] for n in q.get("normalized", [])})
        for page in q.get("pages", {}).values():
            cats = cats_by_title.setdefault(page.get("title"), [])
            cats.extend(c["title"].replace("Category:", "") for c in page.get("categories", []))
        if "continue" in data:
            params.update(data["continue"])
            time.sleep(REQUEST_DELAY_SEC)
        else:
            break

    result = {}
    for t in titles:
        t1 = norm_map.get(t, t)
        final = redirect_map.get(t1, t1)
        result[t] = (final, cats_by_title.get(final, []))
    return result


def get_article_text(title: str):
    """Чистый текст статьи. None = запрос не удался, "" = у статьи нет текста."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "titles": title,
        "format": "json",
    }
    data = api_get(params)
    if data is None:
        return None
    for page in data.get("query", {}).get("pages", {}).values():
        return (page.get("extract") or "").strip()
    return ""


# ======================= ФИЛЬТРЫ И ТЕГИ =======================

def matches_patterns(categories, patterns):
    cats_text = " | ".join(categories).lower()
    return any(re.search(p, cats_text) for p in patterns)


def is_denied(title: str) -> bool:
    return any(re.search(p, title, re.IGNORECASE) for p in DENY_TITLE_PATTERNS)


def infer_tag(title: str, categories, hub_tag: str) -> str:
    """Тип статьи по её категориям; если ничего не подошло - тег хаба."""
    cats = " | ".join(categories).lower()
    t = title.lower()
    if t.startswith("list of") or t.startswith("lists of"):
        return "list"
    if "soundtrack" in cats or "(soundtrack)" in t:
        return "soundtrack"
    if re.search(r"\b(19|20)\d{2}\s+births\b", cats):
        return "actor"
    if "video game" in cats:
        return "video_game"
    if re.search(r"television series|miniseries|web series|streaming television", cats):
        return "tv_series"
    if "marvel cinematic universe" in cats and re.search(r"\bcharacters?\b", cats):
        return "character"
    if re.search(r"\bfilms?\b", cats) and "marvel cinematic universe" in cats:
        return "film"
    return hub_tag


# ======================= СОХРАНЕНИЕ =======================

def safe_filename(title: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", title).strip("_")
    return name[:150]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_article(title: str, tag: str, text: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, safe_filename(title) + ".json")
    new_hash = content_hash(text)

    status = "new"
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("content_hash") == new_hash and existing.get("tag") == tag:
            return "unchanged"
        status = "updated"

    record = {
        "source": "wikipedia",
        "title": title,
        "url": f"https://{LANG}.wikipedia.org/wiki/{title.replace(' ', '_')}",
        "tag": tag,
        "content": text,
        "content_hash": new_hash,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return status


def prune_denied():
    """Удаляет из data/raw статьи, подпадающие под DENY_TITLE_PATTERNS (остались от прошлых запусков)."""
    removed = []
    if not os.path.isdir(OUTPUT_DIR):
        return removed
    for name in os.listdir(OUTPUT_DIR):
        if not name.endswith(".json"):
            continue
        path = os.path.join(OUTPUT_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                title = json.load(f).get("title", "")
        except (OSError, ValueError):
            continue
        if is_denied(title):
            os.remove(path)
            removed.append(title)
    return removed


# ======================= ОСНОВНОЙ ЦИКЛ =======================

class State:
    def __init__(self):
        self.stats = {"new": 0, "updated": 0, "unchanged": 0,
                      "filtered": 0, "denied": 0, "no_text": 0}
        self.seen = set()          # итоговые названия, уже обработанные
        self.failed_cat = []       # (hub, [названия]) - не удалось получить категории
        self.failed_text = []      # (hub, название, категории) - не удалось получить текст


def handle_article(state: State, hub, final_title, cats):
    """Фильтр + скачивание текста + сохранение для одной статьи."""
    if is_denied(final_title):
        state.stats["denied"] += 1
        return
    if not matches_patterns(cats, hub["category_patterns"]):
        state.stats["filtered"] += 1
        return

    text = get_article_text(final_title)
    time.sleep(REQUEST_DELAY_SEC)
    if text is None:
        state.failed_text.append((hub, final_title, cats))
        return
    if not text:
        state.stats["no_text"] += 1
        return

    tag = infer_tag(final_title, cats, hub["tag"])
    status = save_article(final_title, tag, text)
    state.stats[status] += 1
    print(f"    [{status}] ({tag}) {final_title}")


def process_titles(state: State, hub, titles, batch_size=CATEGORY_BATCH):
    for i in range(0, len(titles), batch_size):
        chunk = titles[i:i + batch_size]
        info = get_categories_batch(chunk)
        time.sleep(REQUEST_DELAY_SEC)
        if info is None:
            state.failed_cat.append((hub, chunk))
            continue
        for t in chunk:
            final, cats = info[t]
            if final in state.seen:
                continue
            state.seen.add(final)
            handle_article(state, hub, final, cats)


def run():
    state = State()

    for hub in HUBS:
        print(f"\nХаб: {hub['page']}")
        links = get_links_from_page(hub["page"])
        print(f"  всего ссылок на странице: {len(links)}")
        process_titles(state, hub, links)

    # повторные попытки для упавших запросов
    for attempt in range(1, RETRY_PASSES + 1):
        if not state.failed_cat and not state.failed_text:
            break
        print(f"\nПовторная попытка {attempt}/{RETRY_PASSES}: "
              f"категории - {sum(len(c) for _, c in state.failed_cat)} шт., "
              f"тексты - {len(state.failed_text)} шт.")
        time.sleep(10 * attempt)
        failed_cat, failed_text = state.failed_cat, state.failed_text
        state.failed_cat, state.failed_text = [], []
        for hub, titles in failed_cat:
            process_titles(state, hub, titles, batch_size=10)
        for hub, final, cats in failed_text:
            handle_article(state, hub, final, cats)

    removed = prune_denied()

    s = state.stats
    print("\n===== Готово =====")
    print(f"Новых: {s['new']}")
    print(f"Обновлено: {s['updated']}")
    print(f"Без изменений: {s['unchanged']}")
    print(f"Отфильтровано (не по теме): {s['filtered']}")
    print(f"Отсечено по названию (не MCU): {s['denied']}")
    print(f"Без текста: {s['no_text']}")
    print(f"Удалено старых не-MCU файлов: {len(removed)}")
    lost = sum(len(c) for _, c in state.failed_cat) + len(state.failed_text)
    print(f"НЕ УДАЛОСЬ получить после всех попыток: {lost}")
    if lost:
        print("  -> запусти скрипт ещё раз: уже скачанное будет пропущено, докачается остальное")
    print(f"Файлы сохранены в: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    run()
