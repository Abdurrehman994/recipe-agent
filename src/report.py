"""Summary tables of the data, the scenarios and the evaluation results.

    python -m src.report              # print the tables and save them as CSV in results/summary/
    python -m src.report --no-save
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from src.evaluate import state_items
from src.scenarios import load_scenarios

ROOT = Path(__file__).resolve().parents[1]
RESULTS, LOGS, DATA = ROOT / "results", ROOT / "logs", ROOT / "data" / "processed"
OUT = RESULTS / "summary"
LLM = "-llm-lehmus-gpt-oss-120b"
SPLITS = {"dev": "Dev", "test": "Held-out", "test2": "Held-out 2", "long": "Long"}
CONFIGS = {"rag": "RAG baseline", "rag_memory": "RAG + memory", "agent": "Full agent",
           "agent_nomem": "Agent, no memory", "agent_noval": "Agent, no validation"}


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_log(name: str) -> list[dict]:
    with open(LOGS / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def llm_extraction(split: str) -> str:
    return f"{split}_llm{LLM[len('-llm'):]}"


def fmt(v, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):  # round half up: 0.7525 -> 0.753
        return str(Decimal(str(v)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))
    return str(v)


def show(title: str, rows: list[dict]) -> None:
    cols = list(rows[0])
    cells = [[fmt(r[c]) for c in cols] for r in rows]
    width = [max(len(c), *(len(x[i]) for x in cells)) for i, c in enumerate(cols)]
    print(f"\n{title}")
    print("  ".join(c.ljust(w) for c, w in zip(cols, width)))
    print("  ".join("-" * w for w in width))
    for x in cells:
        print("  ".join(v.ljust(w) for v, w in zip(x, width)))


def save(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def filtering() -> list[dict]:
    rep = json.loads((DATA / "prep_report.json").read_text(encoding="utf-8"))
    return [{"step": "Raw HUMMUS recipe file", "recipes": rep["raw_rows"]},
            {"step": "After duration, tag, nutrition and rating filters", "recipes": rep["after_quality_filters"]},
            {"step": "After ingredient parsing (>= 3 lines)", "recipes": rep["after_ingredient_parse"]},
            {"step": "Stratified sample (60% main dishes)", "recipes": rep["sampled"]}]


def distributions() -> tuple[list[dict], list[dict]]:
    with open(DATA / "recipes.jsonl", encoding="utf-8") as f:
        recipes = [json.loads(line) for line in f]
    time_bins = [("<=15", 0, 15), ("15-30", 15, 30), ("30-45", 30, 45), ("45-60", 45, 60),
                 ("60-90", 60, 90), ("90-120", 90, 120), ("120-180", 120, 180), (">180", 180, 10**9)]
    kcal_bins = [("<200", 0, 200), ("200-400", 200, 400), ("400-600", 400, 600),
                 ("600-800", 600, 800), ("800-1000", 800, 1000), (">=1000", 1000, 10**9)]
    time = [{"minutes": n, "recipes": sum(a < r["duration_min"] <= b for r in recipes)}
            for n, a, b in time_bins]
    kcal = [{"kcal_per_serving": n, "recipes": sum(a <= r["calories_kcal"] < b for r in recipes)}
            for n, a, b in kcal_bins]
    return time, kcal


def scenario_stats() -> list[dict]:
    rows = []
    for split, label in SPLITS.items():
        sc = load_scenarios(split)
        turns = [t for s in sc for t in s["turns"]]
        updates = [u for t in turns for u in t["gold"]["updates"]]
        lengths = [len(s["turns"]) for s in sc]
        rows.append({
            "set": label, "scenarios": len(sc), "user_turns": len(turns),
            "turns_per_scenario": f"{min(lengths)}-{max(lengths)}",
            "recommend": sum(t["expected_action"] == "recommend" for t in turns),
            "clarify": sum(t["expected_action"] == "clarify" for t in turns),
            "report_infeasible": sum(t["expected_action"] == "report_infeasible" for t in turns),
            "competing_objectives": sum(len(t["gold"]["objectives"]) >= 2 for t in turns),
            "resets": sum(t["gold"]["reset"] for t in turns),
            "persistent_updates": sum(u["persistent"] for u in updates),
            "removals": sum(u["op"] == "remove" for u in updates),
        })
    return rows


def extraction() -> list[dict]:
    rows = []
    for split, label in SPLITS.items():
        for name, stem in (("Rules", f"{split}_rules"), ("LLM", llm_extraction(split))):
            res, nomem = load(stem), load(f"{stem}_nomem")
            if res is None:
                continue
            ex, st = res["extraction"], res["state"]
            rows.append({"set": label, "extractor": name, "P": ex["precision"], "R": ex["recall"],
                         "F1": ex["f1"], "state_exact": st["exact_match"],
                         "state_exact_no_memory": nomem["state"]["exact_match"] if nomem else None,
                         "provenance_acc": st["provenance_accuracy"]})
    return rows


def _error_type(item: tuple) -> str:
    if item[0] in ("objective", "reset", "remove"):
        return {"objective": "Objectives", "reset": "Resets", "remove": "Removals"}[item[0]]
    return {"max": "Numeric limits", "min": "Numeric limits", "course": "Courses", "include": "Inclusions",
            "exclude": "Exclusions", "diet": "Diets"}[item[1]]


def extraction_errors(split: str = "test") -> list[dict]:
    counts = {}
    for name, stem in (("rules", f"{split}_rules"), ("llm", llm_extraction(split))):
        c = Counter()
        for turn in load(stem)["per_turn"]:
            for item in turn["extraction_missed"] + turn["extraction_spurious"]:
                c[_error_type(ast.literal_eval(item) if isinstance(item, str) else item)] += 1
        counts[name] = c
    types = ["Objectives", "Courses", "Resets", "Inclusions", "Exclusions", "Diets", "Numeric limits", "Removals"]
    rows = [{"error_type": t, "rules": counts["rules"][t], "llm": counts["llm"][t]} for t in types]
    rows.append({"error_type": "Total", "rules": sum(counts["rules"].values()), "llm": sum(counts["llm"].values())})
    return rows


def state_by_turn(split: str = "long", max_turn: int = 9) -> list[dict]:
    series = {"llm": llm_extraction(split), "rules": f"{split}_rules",
              "llm_memory_off": f"{llm_extraction(split)}_nomem"}
    by = {k: defaultdict(list) for k in series}
    for key, stem in series.items():
        for turn in load(stem)["per_turn"]:
            by[key][turn["turn"]].append(turn["state_exact"])
    return [{"turn": t, "dialogues": len(by["llm"][t]),
             **{k: round(sum(v[t]) / len(v[t]), 3) for k, v in by.items()}}
            for t in range(1, max_turn + 1)]


def failure_attribution(split: str, llm: bool) -> dict:
    gold = {(s["id"], i): t for s in load_scenarios(split) for i, t in enumerate(s["turns"], start=1)}
    name = f"{split}_agent{LLM if llm else ''}"
    res, logs = load(name), {(r["scenario"], r["turn"]): r for r in read_log(name)}
    relax, wrong = [0, 0], [0, 0]  # index 1: the agent's state matched the gold state
    for p in res["per_turn"]:
        g, rec = gold[(p["scenario"], p["turn"])], logs[(p["scenario"], p["turn"])]
        correct = (state_items(g["active"]) == state_items(rec["state"])
                   and {(o["target"], o["direction"]) for o in g["active_objectives"]}
                   == {(o["target"], o["direction"]) for o in rec["objectives"]})
        if p["action"] == "recommend" and (p["violations"] or g["expected_action"] == "report_infeasible"):
            relax[correct] += 1
        if p["action"] != p["expected_action"]:
            wrong[correct] += 1
    return {"action_acc": res["action_accuracy"], "clarify_recall": res["clarify_recall"],
            "infeasible_recall": res["infeasible_recall"],
            "relaxations_state_wrong": relax[0], "relaxations_state_correct": relax[1],
            "wrong_actions_state_wrong": wrong[0], "wrong_actions_state_correct": wrong[1]}


def policy() -> list[dict]:
    return [{"set": SPLITS[split], "extractor": "LLM" if llm else "Rules", **failure_attribution(split, llm)}
            for split in ("test", "test2") for llm in (False, True)]


def system_result(split: str, config: str, llm: bool) -> dict | None:
    res = load(f"{split}_{config}{LLM if llm else ''}")
    if res is None and llm and config == "rag":  # the RAG baseline has no extractor
        res = load(f"{split}_{config}")
    return res


def systems() -> list[dict]:
    rows = []
    for config, label in CONFIGS.items():
        row = {"system": label}
        for split in ("test", "test2"):
            for llm in (False, True):
                r = system_result(split, config, llm)
                tag = f"{split}_{'llm' if llm else 'rules'}"
                row[f"{tag}_satisfaction"] = r["constraint_satisfaction"]
                row[f"{tag}_relaxations"] = r["silent_relaxations"]
        rows.append(row)
    return rows


def tool_usage() -> list[dict]:
    res = load(f"test_agent{LLM}")
    rejected = sum(len(r["rejected"]) for r in read_log(f"test_agent{LLM}"))
    return [{"tool": k, "calls": v} for k, v in res["tool_calls"].items()] + [
        {"tool": "calls per turn", "calls": res["tool_calls_per_turn"]},
        {"tool": "candidates rejected by validation", "calls": rejected}]


def claims() -> list[dict]:
    rows = []
    for split in ("test", "test2", "long"):
        runs = [load(p.stem) for p in sorted(RESULTS.glob(f"{split}_*.json"))
                if not any(x in p.stem for x in ("_pref", "_llmexp"))]
        runs = [r for r in runs if r and "n_claims" in r]
        rows.append({"set": SPLITS[split], "runs": len(runs), "claims": sum(r["n_claims"] for r in runs),
                     "unsupported": sum(round(r["unsupported_claim_rate"] * r["n_claims"]) for r in runs)})
    return rows


def reviewer() -> list[dict]:
    rows = []
    for split in ("dev", "test"):
        rev = load(f"{split}_agent_llmexp_rev{LLM}")["review"]
        rows.append({"set": SPLITS[split], "turns_reviewed": rev["turns_reviewed"],
                     "claims_checked": rev["claims_checked"], "unsupported": rev["unsupported_claims"],
                     "turns_corrected": rev["turns_corrected"]})
    return rows


def preference_ranking() -> list[dict]:
    with open(DATA / "recipes.jsonl", encoding="utf-8") as f:
        db = {r["recipe_id"]: r for r in map(json.loads, f)}
    rows = []
    for extractor, suffix in (("Rules", ""), ("LLM", LLM)):
        for ranking, name in (("popularity", f"test_agent{suffix}"), ("preference", f"test_agent_pref{suffix}")):
            recs = [db[x["recipe_id"]] for r in read_log(name) for x in r["recommendations"]]
            res = load(name)
            rows.append({"extractor": extractor, "ranking": ranking,
                         "satisfaction": res["constraint_satisfaction"], "relaxations": res["silent_relaxations"],
                         "mean_rating": round(sum(r["average_rating"] for r in recs) / len(recs), 2),
                         "mean_kcal": round(sum(r["calories_kcal"] for r in recs) / len(recs)),
                         "mean_minutes": round(sum(r["duration_min"] for r in recs) / len(recs), 1),
                         "share_100_800_kcal": round(sum(100 <= r["calories_kcal"] <= 800 for r in recs) / len(recs), 2)})
    return rows


def all_tables() -> dict[str, list[dict]]:
    time, kcal = distributions()
    return {
        "filtering": filtering(),
        "time_distribution": time,
        "calorie_distribution": kcal,
        "scenarios": scenario_stats(),
        "extraction": extraction(),
        "extraction_errors": extraction_errors(),
        "state_by_turn_long": state_by_turn(),
        "policy": policy(),
        "systems": systems(),
        "tool_usage": tool_usage(),
        "claims": claims(),
        "reviewer": reviewer(),
        "preference_ranking": preference_ranking(),
    }


def main():
    ap = argparse.ArgumentParser(description="Print summary tables of the results.")
    ap.add_argument("--no-save", action="store_true", help="print only")
    args = ap.parse_args()
    tables = all_tables()
    for name, rows in tables.items():
        show(name, rows)
        if not args.no_save:
            save(name, rows)
    if not args.no_save:
        print(f"\nsaved {len(tables)} tables to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
