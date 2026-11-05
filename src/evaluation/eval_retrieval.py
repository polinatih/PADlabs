"""
Оценка retrieval на data/eval/questions.jsonl.

Метрики считаются на уровне СТАТЕЙ (а не chunk_id), чтобы стратегии chunking
с разными chunks можно было сравнивать честно:
  Hit@K    - есть ли среди Top-K chunks хотя бы одна из нужных статей;
  Recall@K - какая доля нужных статей встретилась в Top-K (важно для multi_doc);
  MRR      - 1/позиция первого chunk из нужной статьи (в пределах max K).
Вопросы типа no_answer в метрики не входят: для них считается средний
top-1 score (нужен, чтобы выбрать порог отсечения на этапе filtering).

Запуск:
  pixi run python src/evaluation/eval_retrieval.py
  pixi run python src/evaluation/eval_retrieval.py --strategies headers --models bge --ks 1 3 5 10
Результат: experiments/retrieval_summary.csv и experiments/retrieval_per_question.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "retrieval"))
from retriever import Retriever  # noqa: E402

STRATEGIES = ["fixed", "fixed_overlap", "paragraph", "headers"]


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_question(hits: list[dict], relevant: set[str], k: int) -> dict:
    top = hits[:k]
    found = {h["article_title"] for h in top} & relevant
    rr = 0.0
    for h in hits[:k]:
        if h["article_title"] in relevant:
            rr = 1.0 / h["rank"]
            break
    return {"hit": 1.0 if found else 0.0, "recall": len(found) / len(relevant), "rr": rr}


def evaluate(retriever: Retriever, questions: list[dict], ks: list[int]):
    kmax = max(ks)
    per_q, agg = [], {k: {"hit": [], "recall": [], "rr": []} for k in ks}
    no_ans_scores = []
    for qn in questions:
        hits = retriever.search(qn["question"], k=kmax)
        if qn["type"] == "no_answer" or not qn["relevant_articles"]:
            no_ans_scores.append(hits[0]["score"] if hits else 0.0)
            continue
        rel = set(qn["relevant_articles"])
        for k in ks:
            s = score_question(hits, rel, k)
            for key in s:
                agg[k][key].append(s[key])
            per_q.append({"strategy": retriever.strategy, "model": retriever.model_key, "k": k,
                          "id": qn["id"], "type": qn["type"], **s})
    summary = []
    for k in ks:
        n = len(agg[k]["hit"])
        summary.append({
            "strategy": retriever.strategy, "model": retriever.model_key, "k": k, "n_questions": n,
            "hit_at_k": round(sum(agg[k]["hit"]) / n, 3),
            "recall_at_k": round(sum(agg[k]["recall"]) / n, 3),
            "mrr": round(sum(agg[k]["rr"]) / n, 3),
            "no_answer_top1_score": round(sum(no_ans_scores) / len(no_ans_scores), 3) if no_ans_scores else "",
        })
    return summary, per_q


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--models", nargs="+", default=["minilm", "bge"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--questions", type=Path, default=ROOT / "data" / "eval" / "questions.jsonl")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "experiments")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    all_sum, all_q = [], []
    for model in args.models:
        for strat in args.strategies:
            try:
                r = Retriever(strat, model, data_dir=args.data_dir)
            except Exception as e:  # коллекции может не быть
                print(f"[пропуск] {strat}__{model}: {e}")
                continue
            s, pq = evaluate(r, questions, args.ks)
            all_sum += s
            all_q += pq
            for row in s:
                print(f"{strat:14s} {model:7s} K={row['k']:<3d} Hit={row['hit_at_k']:.3f} "
                      f"Recall={row['recall_at_k']:.3f} MRR={row['mrr']:.3f}")
    write_csv(args.out_dir / "retrieval_summary.csv", all_sum)
    write_csv(args.out_dir / "retrieval_per_question.csv", all_q)
    print(f"\nСохранено в {args.out_dir}")


if __name__ == "__main__":
    main()
