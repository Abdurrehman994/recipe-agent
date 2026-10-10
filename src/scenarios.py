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

from src.schema import Constraint, TurnExtraction
from src.state import DialogueState

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios"
ACTIONS = {"recommend", "clarify", "report_infeasible"}


def load_scenarios(split: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((SCENARIO_DIR / split).glob("*.json"))]


def apply_extraction(active: list[dict], ext: TurnExtraction, turn: int) -> list[dict]:
    """Replay one turn of gold updates with the same rules the system's memory uses."""
    return DialogueState.from_active(active).apply(ext, turn).to_active()


def as_set(active: list[dict]) -> set[Constraint]:
    return {Constraint.from_dict(a) for a in active}


def validate(scenario: dict, db=None) -> list[str]:
    """Return a list of annotation problems (empty = OK). With a db, also checks feasibility."""
    from src.tools import search_recipes  # local import: loading the db is optional
    errors, state = [], DialogueState()
    key = lambda a: (a["kind"], a["target"], a.get("value") or 0)
    for i, turn in enumerate(scenario["turns"], start=1):
        where = f"{scenario['id']} turn {i}"
        try:
            state.apply(TurnExtraction.from_dict(turn["gold"]), i)
            gold = sorted(DialogueState.from_active(turn["active"]).to_active(), key=key)
            gold_obj = sorted((o["target"], o["direction"]) for o in turn["active_objectives"])
        except (KeyError, ValueError, TypeError) as e:
            errors.append(f"{where}: bad annotation: {e!r}")
            continue
        replayed = sorted(state.to_active(), key=key)
        replayed_obj = sorted((o.target, o.direction) for o in state.objectives)
        if gold != replayed or gold_obj != replayed_obj:
            errors.append(f"{where}: 'active' does not match replayed updates\n"
                          f"   annotated: {gold} {gold_obj}\n   replayed:  {replayed} {replayed_obj}")
            # continue from the annotation to avoid cascading errors
            state = DialogueState.from_active(turn["active"], turn["active_objectives"])
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
