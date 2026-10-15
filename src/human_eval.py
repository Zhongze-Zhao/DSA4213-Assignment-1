"""Blind human evaluation of generated text (fluency and consistency, 1-3 scale).

    python -m src.human_eval make      # writes eval/human_eval_sheet.csv (+ hidden key)
    python -m src.human_eval analyze --sheet eval/human_eval_sheet_scored.csv   # after rating

Samples: the top-k continuations from results/generations.json (same prompts, length and decoding
for every model), 2 per prompt and model, drawn at random from samples 1-9 (sample 0 is the one
shown in the report's Table 2, so the rater cannot recognise it). All items are shuffled together
and the sheet shows no model names; the item -> model key is stored separately.
"""
import argparse
import csv
import json

import numpy as np

from src.utils import ROOT, save_json

MODELS = ["ngram", "rnn", "lstm", "transformer", "lstm_w2v", "lstm_glove"]
PER_PROMPT = 2
SEED = 2026
SHEET = ROOT / "eval/human_eval_sheet.csv"
KEY = ROOT / "eval/human_eval_key.json"
FIELDS = ["id", "prompt", "continuation", "fluency (1-3)", "consistency (1-3)", "note"]


def make():
    g = json.load(open(ROOT / "results/generations.json"))
    rng = np.random.default_rng(SEED)
    items = []
    for m in MODELS:
        for pi, prompt in enumerate(g["prompts"]):
            for si in rng.choice(np.arange(1, len(g["models"][m]["sample"][pi])), PER_PROMPT, replace=False):
                items.append({"model": m, "prompt_idx": pi, "sample_idx": int(si), "prompt": prompt,
                              "text": g["models"][m]["sample"][pi][si]})
    order = rng.permutation(len(items))
    SHEET.parent.mkdir(exist_ok=True)
    if SHEET.exists():
        raise SystemExit(f"{SHEET} already exists; delete it first to regenerate (ratings would be lost)")
    key = {}
    # utf-8-sig so that Excel shows the text correctly; the arrow stops Excel from reading a
    # continuation that starts with "=" (WikiText heading markup) as a formula
    with open(SHEET, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for n, i in enumerate(order, 1):
            it = items[i]
            key[n] = {k: it[k] for k in ("model", "prompt_idx", "sample_idx")}
            w.writerow([n, it["prompt"], "→ " + it["text"], "", "", ""])
    json.dump(key, open(KEY, "w"), indent=1)
    print(f"wrote {len(items)} items to {SHEET.relative_to(ROOT)} (key: {KEY.relative_to(ROOT)})")


def boot_ci(x, rng, n=10000):
    means = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze(sheet=SHEET):
    key = json.load(open(KEY))
    rows = list(csv.DictReader(open(ROOT / sheet, encoding="utf-8-sig")))
    scores = {m: {"fluency": [], "consistency": []} for m in MODELS}
    problems = []
    for r in rows:
        for dim, col in (("fluency", "fluency (1-3)"), ("consistency", "consistency (1-3)")):
            v = r[col].strip()
            if v not in {"1", "2", "3"}:
                problems.append(f"item {r['id']}: {col} = {v!r}")
                continue
            scores[key[r["id"]]["model"]][dim].append(int(v))
    if problems:
        print("Missing/invalid scores (fix the sheet and rerun):\n  " + "\n  ".join(problems))
    rng = np.random.default_rng(0)
    out = {}
    for m, d in scores.items():
        out[m] = {}
        for dim, v in d.items():
            if v:
                lo, hi = boot_ci(np.array(v), rng)
                out[m][dim] = {"mean": float(np.mean(v)), "ci95": [lo, hi], "n": len(v),
                               "counts": {s: v.count(s) for s in (1, 2, 3)}}
        print(m, {dim: f"{s['mean']:.2f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}] n={s['n']}"
                  for dim, s in out[m].items()})
    # two-sided Mann-Whitney U tests for the comparisons discussed in the report
    from scipy.stats import mannwhitneyu
    tests = {}
    for a, b in [("transformer", "ngram"), ("transformer", "lstm"), ("transformer", "rnn"),
                 ("lstm_w2v", "lstm"), ("lstm_w2v", "lstm_glove"), ("lstm_glove", "lstm")]:
        for dim in ("fluency", "consistency"):
            p = mannwhitneyu(scores[a][dim], scores[b][dim], alternative="two-sided").pvalue
            tests[f"{a} vs {b} / {dim}"] = float(p)
            print(f"{a} vs {b} ({dim}): p = {p:.3f}")
    save_json({"scores": out, "mann_whitney_p": tests, "n_problems": len(problems), "scale": "1-3",
               "items_per_model": 5 * PER_PROMPT}, "results/human_eval.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "analyze"])
    ap.add_argument("--sheet", default=str(SHEET.relative_to(ROOT)), help="rated copy of the sheet")
    args = ap.parse_args()
    make() if args.cmd == "make" else analyze(args.sheet)