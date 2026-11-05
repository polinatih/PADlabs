"""
Тест grabber'а на имитации Wikipedia API (без интернета).
Проверяет ошибки, которые уже случались в реальных запусках:
  - ссылок больше 300 (раньше хвост алфавита терялся);
  - упавший запрос категорий / текста должен быть повторён, а не потерян;
  - редирект не создаёт дубликат файла;
  - не-MCU статьи не сохраняются, старые такие файлы удаляются;
  - теги определяются по категориям;
  - повторный запуск ничего не меняет.

Запуск:  pixi run -e dev pytest
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "grabber"))

import wiki_grabber as wg  # noqa: E402

MCU_FILM_CAT = "Marvel Cinematic Universe: Phase One films"

FILMS = [f"Film {i:03d}" for i in range(700)]
LINKS = FILMS + [
    "Alias X",                                      # редирект на Film 650
    "Hero (2002 film series character)",            # не MCU, но с MCU-категорией
    "Random Article",                               # вообще не по теме
    "Score (soundtrack)",                           # саундтрек
]
REDIRECTS = {"Alias X": "Film 650"}
CATS = {f: [MCU_FILM_CAT] for f in FILMS}
CATS["Hero (2002 film series character)"] = ["Marvel Cinematic Universe characters"]
CATS["Random Article"] = ["Some other category"]
CATS["Score (soundtrack)"] = ["Marvel Cinematic Universe film soundtracks"]


@pytest.fixture
def fake_wiki(tmp_path, monkeypatch):
    state = {"cat_failed": False, "text_failed": False}

    def fake_api(params):
        prop = params.get("prop")
        if prop == "links":
            start = int(params.get("plcontinue", 0))
            data = {"query": {"pages": {"1": {"links": [{"title": t} for t in LINKS[start:start + 500]]}}}}
            if start + 500 < len(LINKS):
                data["continue"] = {"plcontinue": str(start + 500), "continue": "||"}
            return data

        if prop == "categories":
            titles = params["titles"].split("|")
            if "Film 100" in titles and not state["cat_failed"]:
                state["cat_failed"] = True        # первый запрос с этой пачкой падает
                return None
            pages, redirects = {}, []
            for t in titles:
                final = REDIRECTS.get(t, t)
                if final != t:
                    redirects.append({"from": t, "to": final})
                pages[final] = {"title": final,
                                "categories": [{"title": "Category:" + c} for c in CATS.get(final, [])]}
            return {"query": {"redirects": redirects, "pages": {str(i): p for i, p in enumerate(pages.values())}}}

        if prop == "extracts":
            title = params["titles"]
            if title == "Film 005" and not state["text_failed"]:
                state["text_failed"] = True       # первый запрос текста падает
                return None
            return {"query": {"pages": {"1": {"title": title, "extract": f"Text of {title}"}}}}

        raise AssertionError(f"неожиданный запрос {params}")

    monkeypatch.setattr(wg, "api_get", fake_api)
    monkeypatch.setattr(wg.time, "sleep", lambda s: None)
    monkeypatch.setattr(wg, "OUTPUT_DIR", str(tmp_path / "raw"))
    monkeypatch.setattr(wg, "HUBS", [{"page": "Hub", "category_patterns": [r"marvel cinematic universe"], "tag": "film"}])
    return tmp_path / "raw"


def _titles(raw: Path):
    return {json.loads(p.read_text(encoding="utf-8"))["title"]: json.loads(p.read_text(encoding="utf-8"))
            for p in raw.glob("*.json")}


def test_full_run(fake_wiki):
    # старый файл от прошлого запуска, который должен быть удалён
    fake_wiki.mkdir(parents=True)
    (fake_wiki / "Old.json").write_text(json.dumps({"title": "Peter Parker (2002 film series character)"}), encoding="utf-8")

    wg.run()
    saved = _titles(fake_wiki)

    # 1. все 700 фильмов (раньше обрезалось на 300), включая упавшие запросы
    assert all(f in saved for f in FILMS), "потеряны статьи"
    assert "Film 650" in saved and "Alias X" not in saved   # редирект без дубликата
    assert sum(1 for t in saved if t.startswith("Film ")) == 700

    # 2. не-MCU не сохранены, старый файл удалён
    assert "Hero (2002 film series character)" not in saved
    assert "Peter Parker (2002 film series character)" not in saved
    assert "Random Article" not in saved

    # 3. теги по категориям
    assert saved["Film 000"]["tag"] == "film"
    assert saved["Score (soundtrack)"]["tag"] == "soundtrack"


def test_second_run_is_idempotent(fake_wiki, capsys):
    wg.run()
    capsys.readouterr()
    wg.run()
    out = capsys.readouterr().out
    assert "Новых: 0" in out
    assert "Без изменений: 701" in out    # 700 фильмов + саундтрек
    assert "НЕ УДАЛОСЬ получить после всех попыток: 0" in out
