
"""
Preprocessing + Chunking для лабы 1.
 
ЧТО ДЕЛАЕТ ЭТОТ СКРИПТ (пункты 5-6 из задания):
 
1. Читает все .json файлы из data/raw/ (сырые статьи от grabber'а).
2. Чистит текст:
   - убирает технические остатки Wikipedia (маркеры ссылок вида [1], [2]);
   - убирает лишние пустые строки и пробелы;
   - превращает заголовки секций вида "== Plot ==" в более удобный вид.
3. Разбивает текст на chunks ЧЕТЫРЬМЯ разными способами, потому что
   в задании нужно сравнить стратегии chunking:
   a) fixed        — фиксированный размер, без пересечения
   b) fixed_overlap — фиксированный размер, с пересечением (overlap)
   c) paragraph    — разбиение по абзацам
   d) headers      — разбиение по заголовкам/секциям статьи
4. Для каждой стратегии результат сохраняется в свой файл:
   data/processed/<strategy>.jsonl
   (один chunk = одна строка JSON, с текстом и metadata).

"""
 
import json
import os
import re
import glob
 
 
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
 
FIXED_CHUNK_SIZE = 500       # символов на chunk для стратегии "fixed"
OVERLAP_CHUNK_SIZE = 500     # символов на chunk для стратегии "fixed_overlap"
OVERLAP_SIZE = 100           # сколько символов пересекается между соседними chunks
MIN_CHUNK_LENGTH = 40        # chunks короче этого (в символах) отбрасываем как мусор
 
 
 
def clean_text(raw_text: str) -> str:
   
    text = raw_text
 
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
 
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
 
    text = re.sub(r"\n{3,}", "\n\n", text)
 
    return text.strip()
 
 
 
def chunk_fixed(text: str, size: int):
    
    chunks = []
    start = 0
    text_len = len(text)
 
    while start < text_len:
        end = min(start + size, text_len)
 
        # Если это не конец текста — подвинем границу до ближайшего пробела
        if end < text_len:
            space_pos = text.rfind(" ", start, end)
            if space_pos > start:
                end = space_pos
 
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk)
 
        start = end
 
    return chunks
 
 
def chunk_fixed_overlap(text: str, size: int, overlap: int):
    
    chunks = []
    start = 0
    text_len = len(text)
    step = max(size - overlap, 1)  # защита от зацикливания при overlap >= size
 
    while start < text_len:
        end = min(start + size, text_len)
 
        if end < text_len:
            space_pos = text.rfind(" ", start, end)
            if space_pos > start:
                end = space_pos
 
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk)
 
        start += step
 
    return chunks
 
 
def chunk_by_paragraphs(text: str):
   
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
 
    chunks = []
    buffer = ""
    for para in raw_paragraphs:
        buffer = (buffer + "\n\n" + para).strip() if buffer else para
        if len(buffer) >= MIN_CHUNK_LENGTH:
            chunks.append(buffer)
            buffer = ""
 
    if buffer:  # остаток, если он есть
        chunks.append(buffer)
 
    return chunks
 
 
def chunk_by_headers(text: str):
    
    header_pattern = re.compile(r"^(={2,4})\s*(.+?)\s*\1$", re.MULTILINE)
 
    matches = list(header_pattern.finditer(text))
    if not matches:
        # Если в статье вообще нет заголовков секций     
        return [("intro", text.strip())] if text.strip() else []
 
    sections = []
 
    intro = text[: matches[0].start()].strip()
    if len(intro) >= MIN_CHUNK_LENGTH:
        sections.append(("intro", intro))
 
    for i, match in enumerate(matches):
        section_title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if len(section_text) >= MIN_CHUNK_LENGTH:
            sections.append((section_title, section_text))
 
    return sections
 
 
 
def load_raw_articles():
    articles = []
    for filepath in glob.glob(os.path.join(RAW_DIR, "*.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            articles.append(json.load(f))
    return articles
 
 
def make_chunk_id(article_title: str, strategy: str, index: int) -> str:
    safe_title = re.sub(r"[^\w\-]+", "_", article_title).strip("_")[:80]
    return f"{safe_title}__{strategy}__{index:03d}"
 
 
def process_articles():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    articles = load_raw_articles()
    print(f"Загружено статей из data/raw: {len(articles)}")
 
    output_files = {
        strategy: open(os.path.join(PROCESSED_DIR, f"{strategy}.jsonl"), "w", encoding="utf-8")
        for strategy in ["fixed", "fixed_overlap", "paragraph", "headers"]
    }
 
    stats = {strategy: {"chunks": 0, "total_chars": 0} for strategy in output_files}
 
    for article in articles:
        title = article.get("title", "unknown")
        url = article.get("url", "")
        tag = article.get("tag", "")
 
        cleaned = clean_text(article.get("content", ""))
        if not cleaned:
            continue
 
        # --- fixed ---
        for idx, chunk_text in enumerate(chunk_fixed(cleaned, FIXED_CHUNK_SIZE)):
            record = {
                "chunk_id": make_chunk_id(title, "fixed", idx),
                "article_title": title,
                "article_url": url,
                "tag": tag,
                "strategy": "fixed",
                "chunk_index": idx,
                "text": chunk_text,
            }
            output_files["fixed"].write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["fixed"]["chunks"] += 1
            stats["fixed"]["total_chars"] += len(chunk_text)
 
        # --- fixed_overlap ---
        for idx, chunk_text in enumerate(chunk_fixed_overlap(cleaned, OVERLAP_CHUNK_SIZE, OVERLAP_SIZE)):
            record = {
                "chunk_id": make_chunk_id(title, "fixed_overlap", idx),
                "article_title": title,
                "article_url": url,
                "tag": tag,
                "strategy": "fixed_overlap",
                "chunk_index": idx,
                "text": chunk_text,
            }
            output_files["fixed_overlap"].write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["fixed_overlap"]["chunks"] += 1
            stats["fixed_overlap"]["total_chars"] += len(chunk_text)
 
        # --- paragraph ---
        for idx, chunk_text in enumerate(chunk_by_paragraphs(cleaned)):
            record = {
                "chunk_id": make_chunk_id(title, "paragraph", idx),
                "article_title": title,
                "article_url": url,
                "tag": tag,
                "strategy": "paragraph",
                "chunk_index": idx,
                "text": chunk_text,
            }
            output_files["paragraph"].write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["paragraph"]["chunks"] += 1
            stats["paragraph"]["total_chars"] += len(chunk_text)
 
        # --- headers ---
        for idx, (section, chunk_text) in enumerate(chunk_by_headers(cleaned)):
            record = {
                "chunk_id": make_chunk_id(title, "headers", idx),
                "article_title": title,
                "article_url": url,
                "tag": tag,
                "strategy": "headers",
                "section": section,
                "chunk_index": idx,
                "text": chunk_text,
            }
            output_files["headers"].write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["headers"]["chunks"] += 1
            stats["headers"]["total_chars"] += len(chunk_text)
 
    for f in output_files.values():
        f.close()
 
    print("\n===== Готово =====")
    for strategy, s in stats.items():
        avg_len = s["total_chars"] / s["chunks"] if s["chunks"] else 0
        print(f"{strategy:15s}: {s['chunks']:5d} chunks, средняя длина {avg_len:.0f} символов "
              f"-> data/processed/{strategy}.jsonl")
 
 
if __name__ == "__main__":
    process_articles()
 
