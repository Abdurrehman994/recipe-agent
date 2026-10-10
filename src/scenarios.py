"""Load and validate annotated dialogue scenarios.

Scenario file (scenarios/<split>/<id>.json):
{
  "id": "dev_001", "split": "dev", "phenomena": ["add", "replace"],
  "turns": [
    {
      "user": "Suggest a vegetarian dinner.",
      "gold": {"updates": [{"op": "add", "constraint": {...}, "persistent": false}, ...],
               "objectives": [], "reset": false},
      "active": [{"kind": "diet", "target": "vegetarian", "source_turn": 1, "persistent": false}],
      "active_objectives": [],
      "expected_action": "recommend"        # recommend | clarify | report_infeasible
    }, ...
  ]
}
Turns are scripted: the next user turn is fixed whatever the agent says, which keeps the
evaluation reproducible. A "clarify" turn is followed by a user turn that answers it.

Usage:  python -m src.scenarios [split]   -> validates files and prints feasibility per turn
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.schema import SINGLE_VALUE_KINDS, Constraint, TurnExtraction

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios"
ACTIONS = {"recommend", "clarify", "report_infeasible"}


def load_scenarios(split: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((SCENARIO_DIR / split).glob("*.json"))]


def apply_extraction(active: list[dict], ext: TurnExtraction, turn: int) -> list[dict]:
    """Reference state update used to check annotations:
    reset drops temporary constraints, add replaces the same slot (max/min/course) or an
    identical constraint, remove drops by slot. Provenance = turn of the latest change."""
    state = [a for a in active if a["persistent"]] if ext.reset else list(active)
    for u in ext.updates:
        c = u.constraint
        def same(a):
            if (a["kind"], a["target"]) != c.slot:
                return False
            return c.kind in SINGLE_VALUE_KINDS or u.op == "remove" or a.get("value") == c.value
        # include/exclude of the same ingredient are mutually exclusive
        flip = {"include": "exclude", "exclude": "include"}.get(c.kind)
        state = [a for a in state if not same(a)
                 and not (u.op == "add" and flip and (a["kind"], a["target"]) == (flip, c.target))]
        if u.op == "add":
            state.append({**c.to_dict(), "source_turn": turn, "persistent": u.persistent})
    return state


def as_set(active: list[dict]) -> set[Constraint]:
    return {Constraint.from_dict(a) for a in active}


def validate(scenario: dict, db=None) -> list[str]:
    """Return a list of annotation problems (empty = OK). With a db, also checks feasibility."""
    from src.tools import search_recipes  # local import: loading the db is optional
    errors, state = [], []
    for i, turn in enumerate(scenario["turns"], start=1):
        where = f"{scenario['id']} turn {i}"
        try:
            ext = TurnExtraction.from_dict(turn["gold"])
            state = apply_extraction(state, ext, i)
            gold = [{**Constraint.from_dict(a).to_dict(), "source_turn": a["source_turn"],
                     "persistent": a["persistent"]} for a in turn["active"]]
        except (KeyError, ValueError) as e:
            errors.append(f"{where}: bad annotation: {e}")
            continue
        key = lambda a: (a["kind"], a["target"], a.get("value") or 0)
        if sorted(gold, key=key) != sorted(state, key=key):
            errors.append(f"{where}: 'active' does not match replayed updates\n"
                          f"   annotated: {sorted(gold, key=key)}\n   replayed:  {sorted(state, key=key)}")
            state = gold  # continue from the annotation to avoid cascading errors
        if turn["expected_action"] not in ACTIONS:
            errors.append(f"{where}: unknown expected_action {turn['expected_action']}")
        if db is not None:
            n = search_recipes(db, constraints=as_set(gold), k=0)["total_matches"]
            turn["_feasible_count"] = n
            if turn["expected_action"] == "recommend" and n == 0:
                errors.append(f"{where}: expected recommend but 0 recipes match")
            if turn["expected_action"] == "report_infeasible" and n > 0:
                errors.append(f"{where}: expected report_infeasible but {n} recipes match")
    return errors


def main(split: str = "dev"):
    from src.tools import RecipeDB
    db = RecipeDB.load()
    scenarios = load_scenarios(split)
    problems = 0
    for sc in scenarios:
        errors = validate(sc, db)
        problems += len(errors)
        print(f"\n{sc['id']}  {', '.join(sc.get('phenomena', []))}")
        for i, t in enumerate(sc["turns"], start=1):
            active = "; ".join(Constraint.from_dict(a).describe() for a in t["active"])
            print(f"  {i}. {t['user'][:60]:<60} -> {t['expected_action']:<17} "
                  f"matches={t.get('_feasible_count', '?'):<5} [{active}]")
        for e in errors:
            print("  !!", e)
    print(f"\n{len(scenarios)} scenarios, {problems} problems")
    return problems


if __name__ == "__main__":
    sys.exit(1 if main(*sys.argv[1:]) else 0)
