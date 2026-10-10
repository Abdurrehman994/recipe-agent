"""Write scenarios in a compact text format and convert them to annotated JSON.

    python -m src.author scenarios/drafts/dev.txt

Format (one file can hold many scenarios):

    ## dev_006 | add, replace | optional note
    > What can I make for breakfast with eggs?
    + course breakfast
    + include egg
    > Nothing that takes more than 20 minutes.
    + max duration_min 20
    > I'm vegan.
    + diet vegan P                  (P = persistent / standing preference)
    > Drop the time limit.
    - max duration_min              (remove; value optional)
    > Forget that, a salad instead.
    reset
    + course salads
    obj min duration_min            (objective: lowest duration)
    = clarify                       (expected action; default recommend)

The `active` state after each turn is computed with the memory rules in src/state.py, then
the validator checks feasibility against the database. Review the printed table: it shows
the state after every turn, so annotation mistakes are easy to spot.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.scenarios import SCENARIO_DIR, validate
from src.schema import Constraint, ConstraintUpdate, Objective, TurnExtraction
from src.state import DialogueState


def parse(text: str) -> list[dict]:
    scenarios, sc, turn = [], None, None
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") and not line.startswith("##"):
            continue
        try:
            if line.startswith("##"):
                parts = [p.strip() for p in line[2:].split("|")]
                sc = {"id": parts[0], "split": parts[0].split("_")[0],
                      "phenomena": [p.strip() for p in parts[1].split(",")] if len(parts) > 1 else [],
                      "notes": parts[2] if len(parts) > 2 else "", "turns": []}
                scenarios.append(sc)
            elif line.startswith(">"):
                turn = {"user": line[1:].strip(), "gold": {"updates": [], "objectives": [], "reset": False},
                        "expected_action": "recommend"}
                sc["turns"].append(turn)
            elif line.startswith(("+", "-")):
                tokens = line[1:].split()
                persistent = tokens[-1] == "P"
                tokens = tokens[:-1] if persistent else tokens
                kind, target = tokens[0], tokens[1]
                value = float(tokens[2]) if len(tokens) > 2 else (0.0 if kind in ("max", "min") else None)
                c = Constraint(kind, target, value)
                turn["gold"]["updates"].append(
                    ConstraintUpdate("add" if line[0] == "+" else "remove", c, persistent).to_dict())
            elif line.startswith("obj"):
                _, direction, target = line.split()
                turn["gold"]["objectives"].append(Objective(target, direction).to_dict())
            elif line == "reset":
                turn["gold"]["reset"] = True
            elif line.startswith("="):
                turn["expected_action"] = line[1:].strip()
            else:
                raise ValueError("unrecognized line")
        except (ValueError, IndexError, AttributeError, TypeError) as e:
            raise SystemExit(f"line {n}: {raw!r}: {e}")
    return scenarios


def fill_active(sc: dict) -> dict:
    state = DialogueState()
    for i, turn in enumerate(sc["turns"], start=1):
        state.apply(TurnExtraction.from_dict(turn["gold"]), i)
        turn["active"] = state.to_active()
        turn["active_objectives"] = [o.to_dict() for o in state.objectives]
    return sc


def main(path: str):
    from src.tools import RecipeDB
    from src.scenarios import main as report
    scenarios = [fill_active(sc) for sc in parse(Path(path).read_text(encoding="utf-8"))]
    db = RecipeDB.load()
    bad = 0
    for sc in scenarios:
        errors = validate(json.loads(json.dumps(sc)), db)
        if errors:
            bad += 1
            print(f"{sc['id']}:\n  " + "\n  ".join(errors))
        out = SCENARIO_DIR / sc["split"] / f"{sc['id']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(sc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(scenarios)} scenarios ({bad} with problems)")
    report(scenarios[0]["split"]) if scenarios else None


if __name__ == "__main__":
    main(sys.argv[1])
