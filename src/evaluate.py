"""Evaluation of constraint extraction and dialogue state.

    python -m src.evaluate --split dev --extractor rules [--no-memory] [--errors]

Per scenario, the extractor reads each user turn and the dialogue state is updated from the
extractor's own output, so extraction errors propagate as in a real conversation.

Metrics
  extraction  micro precision / recall / F1 over atomic items per turn:
              ("add", kind, target, value), ("remove", kind, target), ("objective", target,
              direction), ("reset",). Removals ignore the value.
              persistence accuracy = correct persistent flag among correctly extracted adds.
  state       exact-match accuracy per turn (constraints + objectives identical to gold),
              constraint-level micro P / R / F1, and provenance accuracy (source_turn and
              persistent flag correct among correctly predicted constraints).

--no-memory is the memory ablation: the state is rebuilt from the current turn alone.
Results are written to results/<split>_<extractor>[_nomem].json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.scenarios import load_scenarios
from src.schema import TurnExtraction
from src.state import DialogueState

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def extraction_items(ext: TurnExtraction) -> set[tuple]:
    items = set()
    for u in ext.updates:
        c = u.constraint
        items.add(("add", c.kind, c.target, c.value) if u.op == "add" else ("remove", c.kind, c.target))
    items |= {("objective", o.target, o.direction) for o in ext.objectives}
    if ext.reset:
        items.add(("reset",))
    return items


def state_items(active: list[dict]) -> set[tuple]:
    return {(a["kind"], a["target"], float(a["value"]) if a.get("value") is not None else None)
            for a in active}


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def run(extractor, scenarios: list[dict], memory: bool = True) -> dict:
    ex = {"tp": 0, "fp": 0, "fn": 0}
    st = {"tp": 0, "fp": 0, "fn": 0}
    persist_ok = persist_n = prov_ok = prov_n = exact = turns = 0
    per_turn = []
    for sc in scenarios:
        state = DialogueState()
        for i, turn in enumerate(sc["turns"], start=1):
            gold_ext = TurnExtraction.from_dict(turn["gold"])
            pred_ext = extractor.extract(turn["user"], state)
            if not memory:
                state = DialogueState()
            state.apply(pred_ext, i)

            # extraction
            g, p = extraction_items(gold_ext), extraction_items(pred_ext)
            ex["tp"] += len(g & p); ex["fp"] += len(p - g); ex["fn"] += len(g - p)
            gold_persist = {(u.constraint, u.persistent) for u in gold_ext.updates if u.op == "add"}
            for u in pred_ext.updates:
                match = [gp for c, gp in gold_persist if c == u.constraint]
                if u.op == "add" and match:
                    persist_n += 1
                    persist_ok += match[0] == u.persistent

            # state
            gs, ps = state_items(turn["active"]), state_items(state.to_active())
            st["tp"] += len(gs & ps); st["fp"] += len(ps - gs); st["fn"] += len(gs - ps)
            gold_obj = {(o["target"], o["direction"]) for o in turn["active_objectives"]}
            pred_obj = {(o.target, o.direction) for o in state.objectives}
            ok = gs == ps and gold_obj == pred_obj
            exact += ok
            turns += 1
            gold_prov = {(a["kind"], a["target"], float(a["value"]) if a.get("value") is not None
                          else None): (a["source_turn"], a["persistent"]) for a in turn["active"]}
            for a in state.to_active():
                key = (a["kind"], a["target"], a.get("value"))
                if key in gold_prov:
                    prov_n += 1
                    prov_ok += gold_prov[key] == (a["source_turn"], a["persistent"])

            per_turn.append({
                "scenario": sc["id"], "turn": i, "user": turn["user"],
                "extraction_missed": sorted(map(str, g - p)), "extraction_spurious": sorted(map(str, p - g)),
                "state_exact": ok, "state_missing": sorted(map(str, gs - ps)),
                "state_spurious": sorted(map(str, ps - gs)),
                "objectives_gold": sorted(gold_obj), "objectives_pred": sorted(pred_obj),
            })
    return {
        "extractor": getattr(extractor, "name", type(extractor).__name__), "memory": memory,
        "n_scenarios": len(scenarios), "n_turns": turns,
        "extraction": {**prf(**ex),
                       "persistence_accuracy": round(persist_ok / persist_n, 4) if persist_n else None},
        "state": {"exact_match": round(exact / turns, 4) if turns else None, **prf(**st),
                  "provenance_accuracy": round(prov_ok / prov_n, 4) if prov_n else None},
        "per_turn": per_turn,
    }


def get_extractor(name: str, provider: str = "anthropic", model: str | None = None,
                  effort: str = "medium", fresh: bool = False):
    if name == "rules":
        from src.extract_rules import RuleExtractor
        return RuleExtractor()
    if name == "llm":
        from src.extract_llm import LLMExtractor
        return LLMExtractor(provider=provider, model=model, effort=effort, fresh=fresh)
    if name == "gold":  # upper bound / sanity check: replays the annotations
        return GoldExtractor()
    raise ValueError(f"unknown extractor {name}")


class GoldExtractor:
    """Returns the gold extraction for a known utterance. Used to sanity-check the harness."""
    name = "gold"

    def __init__(self, scenarios=None):
        self.lookup = {}
        for split in ("dev", "test"):
            for sc in scenarios or load_scenarios(split):
                for t in sc["turns"]:
                    self.lookup[t["user"]] = TurnExtraction.from_dict(t["gold"])

    def extract(self, utterance, state=None):
        return self.lookup[utterance]


def print_summary(res: dict):
    e, s = res["extraction"], res["state"]
    print(f"\n{res['extractor']}  memory={res['memory']}  "
          f"({res['n_scenarios']} scenarios, {res['n_turns']} turns)")
    print(f"  extraction  P={e['precision']:.3f}  R={e['recall']:.3f}  F1={e['f1']:.3f}  "
          f"persistence acc={e['persistence_accuracy']}")
    print(f"  state       exact={s['exact_match']:.3f}  P={s['precision']:.3f}  R={s['recall']:.3f}  "
          f"F1={s['f1']:.3f}  provenance acc={s['provenance_accuracy']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--extractor", default="rules", help="rules | llm | gold")
    ap.add_argument("--provider", default="anthropic", help="llm only: anthropic | groq | lehmus")
    ap.add_argument("--model", default=None, help="llm only: override the provider's default model")
    ap.add_argument("--effort", default="medium", help="llm only: reasoning effort")
    ap.add_argument("--fresh", action="store_true", help="llm only: ignore cached answers")
    ap.add_argument("--scenarios", type=int, default=None, help="use only the first N scenarios")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--errors", action="store_true", help="print turns with mistakes")
    args = ap.parse_args()

    extractor = get_extractor(args.extractor, args.provider, args.model, args.effort, args.fresh)
    scenarios = load_scenarios(args.split)[: args.scenarios]
    res = run(extractor, scenarios, memory=not args.no_memory)
    if hasattr(extractor, "dropped"):  # LLM outputs that failed schema validation
        res["dropped"] = extractor.dropped
    if hasattr(extractor, "usage_summary"):
        res["usage"] = extractor.usage_summary()
    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = ("_nomem" if args.no_memory else "") + (f"_first{args.scenarios}" if args.scenarios else "")
    out = RESULTS_DIR / f"{args.split}_{extractor.name}{suffix}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print_summary(res)
    if res.get("usage"):
        u = res["usage"]
        print(f"  llm usage: {u['calls']} calls ({u['api_calls']} api, {u['cached_calls']} cached), "
              f"{u['total_tokens']} tokens, {u['tokens_per_call']} per call "
              f"(prompt {u['prompt_tokens']}, completion {u['completion_tokens']})")
    if res.get("dropped"):
        print(f"  dropped {len(res['dropped'])} invalid LLM items (see 'dropped' in the results file)")
    if args.errors:
        for t in res["per_turn"]:
            if t["extraction_missed"] or t["extraction_spurious"] or not t["state_exact"]:
                print(f"\n  {t['scenario']} t{t['turn']}: {t['user']}")
                for k in ("extraction_missed", "extraction_spurious", "state_missing", "state_spurious"):
                    if t[k]:
                        print(f"     {k}: {t[k]}")
                if t["objectives_gold"] != t["objectives_pred"]:
                    print(f"     objectives gold={t['objectives_gold']} pred={t['objectives_pred']}")
    print(f"\nsaved {out.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
