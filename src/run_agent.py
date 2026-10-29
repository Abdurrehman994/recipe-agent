"""Run a system configuration over scripted scenarios; write conversation logs and metrics.

    python -m src.run_agent --split dev --mode agent [--no-memory] [--no-validate]
                            [--extractor rules|llm] [--provider groq] [--k 3]

Outputs
  logs/<split>_<config>[-<extractor>].jsonl   one line per turn: state, tool calls, action,
                                               recommendations, claims, message
  results/<split>_<config>[-<extractor>].json  metrics + per-turn details

Metrics (spec item 8)
  action_accuracy         predicted action == expected_action, over all turns
  clarify_recall          clarify turns answered with a clarification
  infeasible_recall       infeasible turns answered with report_infeasible
  silent_relaxation_rate  turns where the system recommended although no recipe satisfies the
                          gold active constraints (the thing the agent must never do)
  constraint_satisfaction recommended recipes that satisfy ALL gold active constraints, over
                          all recommended recipes
  turn_all_satisfied      recommend turns where every recommendation satisfies the gold state
  unsupported_claim_rate  factual claims in messages not supported by the database
  state_exact_match       predicted dialogue state == gold (rag has no state: reported as None)
  tool_calls_per_turn     and counts per tool
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.agent import Agent, verify_claims
from src.evaluate import state_items
from src.scenarios import load_scenarios
from src.schema import Constraint
from src.tools import RecipeDB, satisfies_all

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR, RESULTS_DIR = ROOT / "logs", ROOT / "results"


def run(agent: Agent, scenarios: list[dict], db: RecipeDB) -> tuple[dict, list[dict]]:
    records, per_turn = [], []
    n = Counter()
    tool_counts = Counter()
    for sc in scenarios:
        agent.reset_dialogue()
        for i, turn in enumerate(sc["turns"], start=1):
            rec = agent.turn(turn["user"])
            rec = {"scenario": sc["id"], **rec}
            records.append(rec)
            gold_action = turn["expected_action"]
            gold_cs = [Constraint.from_dict(a) for a in turn["active"]]
            feasible = any(satisfies_all(r, gold_cs) for r in db.recipes.values()) if gold_cs else True
            recs = rec["recommendations"]
            sat = [satisfies_all(db.recipes[r["recipe_id"]], gold_cs) for r in recs]
            unsupported = verify_claims(db, rec["claims"])

            n["turns"] += 1
            n["action_ok"] += rec["action"] == gold_action
            n[f"gold_{gold_action}"] += 1
            n[f"hit_{gold_action}"] += rec["action"] == gold_action
            if rec["action"] == "recommend":
                n["recommend_turns"] += 1
                n["recs"] += len(recs)
                n["recs_ok"] += sum(sat)
                n["turn_all_ok"] += bool(recs) and all(sat)
                n["silent_relax"] += (not feasible) or (bool(recs) and not all(sat))
            n["claims"] += len(rec["claims"])
            n["unsupported"] += len(unsupported)
            for r in recs:  # recommendation quality (preference-ranking extension)
                rr = db.recipes[r["recipe_id"]]
                n["q_rating"] += rr["average_rating"]
                n["q_kcal"] += rr["calories_kcal"]
                n["q_minutes"] += rr["duration_min"]
                n["q_sane"] += 100 <= rr["calories_kcal"] <= 800
            if rec.get("review") is not None:  # reviewer extension
                rv = rec["review"]
                n["rev_turns"] += 1
                n["rev_checked"] += rv["n_checked"]
                n["rev_unsupported"] += len(rv["unsupported"])
                n["rev_turns_flagged"] += bool(rv["unsupported"]) or rv.get("no_recipe_named", False)
                n["rev_corrected"] += bool(rv.get("corrected"))
            if agent.mode != "rag":
                n["state_turns"] += 1
                gold_obj = {(o["target"], o["direction"]) for o in turn["active_objectives"]}
                pred_obj = {(o["target"], o["direction"]) for o in rec["objectives"]}
                n["state_ok"] += state_items(turn["active"]) == state_items(rec["state"]) \
                    and gold_obj == pred_obj
            for call in rec["tool_calls"]:
                tool_counts[call["tool"]] += 1
            per_turn.append({
                "scenario": sc["id"], "turn": i, "user": turn["user"],
                "expected_action": gold_action, "action": rec["action"],
                "recommendations": [r["title"] for r in recs], "all_satisfied": bool(recs) and all(sat),
                "violations": [r["title"] for r, ok in zip(recs, sat) if not ok],
                "relaxed": rec["relaxed"], "unsupported_claims": unsupported,
                "n_tool_calls": len(rec["tool_calls"]),
                "review_unsupported": [c["text"] for c in (rec.get("review") or {}).get("unsupported", [])],
                "review_corrected": bool((rec.get("review") or {}).get("corrected")),
            })

    def rate(a, b):
        return round(n[a] / n[b], 4) if n[b] else None

    metrics = {
        "config": agent.name, "extractor": getattr(agent.extractor, "name", None),
        "n_scenarios": len(scenarios), "n_turns": n["turns"],
        "action_accuracy": rate("action_ok", "turns"),
        "clarify_recall": rate("hit_clarify", "gold_clarify"),
        "infeasible_recall": rate("hit_report_infeasible", "gold_report_infeasible"),
        "recommend_recall": rate("hit_recommend", "gold_recommend"),
        "silent_relaxation_rate": rate("silent_relax", "recommend_turns"),
        "silent_relaxations": n["silent_relax"],
        "constraint_satisfaction": rate("recs_ok", "recs"),
        "turn_all_satisfied": rate("turn_all_ok", "recommend_turns"),
        "unsupported_claim_rate": rate("unsupported", "claims"),
        "n_claims": n["claims"],
        "state_exact_match": rate("state_ok", "state_turns"),
        "tool_calls_per_turn": round(sum(tool_counts.values()) / n["turns"], 2) if n["turns"] else 0,
        "tool_calls": dict(sorted(tool_counts.items())),
        "llm_usage": agent.extractor.usage_summary() if hasattr(agent.extractor, "usage_summary") else None,
        "recommendation_quality": {
            "mean_rating": round(n["q_rating"] / n["recs"], 3) if n["recs"] else None,
            "mean_calories_kcal": round(n["q_kcal"] / n["recs"], 1) if n["recs"] else None,
            "mean_duration_min": round(n["q_minutes"] / n["recs"], 1) if n["recs"] else None,
            "share_100_to_800_kcal": rate("q_sane", "recs"),
        },
        "review": None if not n["rev_turns"] else {
            "explanation": agent.explain, "fallback_on_unsupported": agent.review,
            "turns_reviewed": n["rev_turns"], "claims_checked": n["rev_checked"],
            "unsupported_claims": n["rev_unsupported"],
            "unsupported_claim_rate": rate("rev_unsupported", "rev_checked"),
            "turns_flagged": n["rev_turns_flagged"], "turns_corrected": n["rev_corrected"],
            "explainer_usage": agent.llm.usage_summary() if agent.llm is not None
            and agent.llm is not agent.extractor and hasattr(agent.llm, "usage_summary") else None,
        },
        "per_turn": per_turn,
    }
    return metrics, records


def print_summary(m: dict) -> None:
    print(f"\n{m['config']}  extractor={m['extractor']}  ({m['n_scenarios']} scenarios, {m['n_turns']} turns)")
    print(f"  action acc={m['action_accuracy']}  clarify={m['clarify_recall']}  "
          f"infeasible={m['infeasible_recall']}  recommend={m['recommend_recall']}")
    print(f"  constraint satisfaction={m['constraint_satisfaction']}  turns all ok={m['turn_all_satisfied']}  "
          f"silent relaxations={m['silent_relaxations']} ({m['silent_relaxation_rate']})")
    print(f"  unsupported claims={m['unsupported_claim_rate']} of {m['n_claims']}  "
          f"state exact={m['state_exact_match']}  tools/turn={m['tool_calls_per_turn']} {m['tool_calls']}")
    if m.get("llm_usage"):
        u = m["llm_usage"]
        print(f"  llm usage: {u['calls']} calls ({u['api_calls']} api, {u['cached_calls']} cached), "
              f"{u['total_tokens']} tokens, {u['tokens_per_call']} per call")
    q = m.get("recommendation_quality") or {}
    if q.get("mean_rating") is not None:
        print(f"  recommendations: mean rating={q['mean_rating']}  mean kcal={q['mean_calories_kcal']}  "
              f"mean min={q['mean_duration_min']}  share 100-800 kcal={q['share_100_to_800_kcal']}")
    if m.get("review"):
        r = m["review"]
        print(f"  reviewer ({r['explanation']} explanations): {r['claims_checked']} claims checked, "
              f"{r['unsupported_claims']} unsupported ({r['unsupported_claim_rate']}), "
              f"{r['turns_flagged']} turns flagged, {r['turns_corrected']} corrected")


def build_agent(args, db: RecipeDB) -> Agent:
    from src.evaluate import get_extractor
    extractor = get_extractor(args.extractor, args.provider, args.model, args.effort, args.fresh)
    llm = None
    if args.explain == "llm":
        llm = extractor if hasattr(extractor, "complete") else \
            get_extractor("llm", args.provider, args.model, args.effort, args.fresh)
    return Agent(db, extractor, mode=args.mode, memory=not args.no_memory,
                 validate=not args.no_validate, k=args.k, explain=args.explain,
                 review=args.review, ranking=args.ranking, llm=llm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--mode", default="agent", choices=["rag", "rag_memory", "agent"])
    ap.add_argument("--no-memory", action="store_true", help="agent: memory ablation")
    ap.add_argument("--no-validate", action="store_true", help="agent: validation ablation")
    ap.add_argument("--extractor", default="rules", help="rules | llm")
    ap.add_argument("--provider", default="anthropic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--explain", default="template", choices=["template", "llm"],
                    help="extension: model-written explanations (reviewed against tool output)")
    ap.add_argument("--review", action="store_true",
                    help="extension: fall back to the templated answer when the reviewer finds an unsupported claim")
    ap.add_argument("--ranking", default="popularity", choices=["popularity", "preference"],
                    help="extension: preference-based ranking inside the feasible set")
    ap.add_argument("--fresh", action="store_true", help="llm only: ignore cached answers")
    ap.add_argument("--scenarios", type=int, default=None, help="use only the first N scenarios")
    ap.add_argument("--errors", action="store_true", help="print turns with a wrong action or violation")
    args = ap.parse_args()

    db = RecipeDB.load()
    agent = build_agent(args, db)
    metrics, records = run(agent, load_scenarios(args.split)[: args.scenarios], db)
    tag = agent.name + ("" if args.extractor == "rules" else f"-{agent.extractor.name}") \
        + (f"_first{args.scenarios}" if args.scenarios else "")
    LOG_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{args.split}_{tag}.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out = RESULTS_DIR / f"{args.split}_{tag}.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print_summary(metrics)
    print(f"  logs -> {log_path}\n  results -> {out}")
    if args.errors:
        for t in metrics["per_turn"]:
            if t["action"] != t["expected_action"] or t["violations"] or t["unsupported_claims"]:
                print(f"\n  {t['scenario']} t{t['turn']}: {t['user']}")
                print(f"    expected {t['expected_action']}, got {t['action']}; "
                      f"violations={t['violations']} relaxed={t['relaxed']} "
                      f"unsupported={len(t['unsupported_claims'])}")


if __name__ == "__main__":
    main()
