
import requests
import json
import hashlib
import time
import re
import os
from datetime import datetime, timezone


LANG = "en"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
REQUEST_DELAY_SEC = 0.5          
MAX_RETRIES = 4                 
MAX_LINKS_PER_HUB = 300

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

API_URL = f"https://{LANG}.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "LabRAGGrabber/3.0 (student project; contact: example@example.com)"
}


def api_get(params: dict):

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)

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

    print("  [ошибка] не удалось получить ответ после всех попыток, пропускаю")
    return None


def get_links_from_page(title: str):
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
            break

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            for link in page.get("links", []):
                links.append(link["title"])

        if len(links) >= MAX_LINKS_PER_HUB:
            links = links[:MAX_LINKS_PER_HUB]
            break

        if "continue" in data:
            params.update(data["continue"])
            time.sleep(REQUEST_DELAY_SEC)
        else:
            break

    return links


def get_categories(title: str):
    params = {
        "action": "query",
        "prop": "categories",
        "titles": title,
        "cllimit": "50",
        "format": "json",
    }
    data = api_get(params)
    if not data:
        return []

    categories = []
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        for cat in page.get("categories", []):
            categories.append(cat["title"].replace("Category:", ""))
    return categories


def matches_patterns(categories, patterns):
    cats_text = " | ".join(categories).lower()
    return any(re.search(p, cats_text) for p in patterns)


def get_article_text(title: str):
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "titles": title,
        "format": "json",
    }
    data = api_get(params)
    if not data:
        return None

    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract", "").strip()
        if text:
            return text
    return None


def safe_filename(title: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", title).strip("_")
    return name[:150]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_article(title: str, tag: str, text: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = safe_filename(title) + ".json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    new_hash = content_hash(text)

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("content_hash") == new_hash:
            return "unchanged"
        status = "updated"
    else:
        status = "new"

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


def run():
    stats = {"new": 0, "updated": 0, "unchanged": 0, "skipped_filtered": 0, "skipped_no_text": 0}
    seen_titles = set()

    for hub in HUBS:
        hub_page = hub["page"]
        patterns = hub["category_patterns"]
        tag = hub["tag"]

        print(f"\nХаб: {hub_page}")
        candidate_links = get_links_from_page(hub_page)
        print(f"  всего ссылок на странице: {len(candidate_links)}")

        for title in candidate_links:
            if title in seen_titles:
                continue
            seen_titles.add(title)

            categories = get_categories(title)
            time.sleep(REQUEST_DELAY_SEC)

            if not matches_patterns(categories, patterns):
                stats["skipped_filtered"] += 1
                continue

            text = get_article_text(title)
            time.sleep(REQUEST_DELAY_SEC)

            if not text:
                stats["skipped_no_text"] += 1
                continue

            status = save_article(title, tag, text)
            stats[status] += 1
            print(f"    [{status}] ({tag}) {title}")

    print(f"Новых: {stats['new']}")
    print(f"Обновлено: {stats['updated']}")
    print(f"Без изменений: {stats['unchanged']}")
    print(f"Отфильтровано (не по теме): {stats['skipped_filtered']}")
    print(f"Пропущено (нет текста): {stats['skipped_no_text']}")
    print(f"Файлы сохранены в: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    run()