"""Optional extension: model-written explanations and a reviewer that checks them.

    evidence = build_evidence(...)                 # what the tools established this turn
    text = llm.complete(EXPLAIN_SYSTEM, json.dumps(evidence))
    report = review(text, evidence, db)            # every number and claim in `text` re-checked

The reviewer is deterministic code. It finds every quantity in the explanation ("25 min",
"420 kcal", "31 g protein", "9 recipes"), attributes it to the recipe named just before it, and
checks it against the recipe's real value (tolerating rounding), the user's own requirement
values, or the diagnosis of an impossible request. It also checks "meets all your requirements"
statements with the constraint checker. Anything it cannot support is reported, and the agent
can fall back to the templated answer (review=True).
"""
from __future__ import annotations

import re

from src.schema import NUMERIC_FIELDS, Constraint
from src.tools import RecipeDB, satisfies_all

EXPLAIN_SYSTEM = """You write the assistant's reply in a recipe-recommendation chat. You receive JSON \
evidence produced by tools: the user's active requirements, the recommended recipes with their \
verified facts (duration in minutes; calories, protein and other nutrients per serving) and the \
per-requirement checks, an optional trade-off between objectives, or a diagnosis when no recipe \
matches.

Write 2 to 4 plain sentences for the user. Rules:
- Use only numbers that appear in the evidence, quoted exactly with their units, e.g. "25 min", \
"420 kcal", "31 g protein", "8.9 g fat", "640 mg sodium".
- Name recipes exactly as given. Do not invent recipes, ingredients or facts.
- Recommend: name the first recipe, give its duration, calories and protein, say it meets every \
requirement and mention the other verified options briefly.
- Trade-off: name the best recipe for each objective and give the compared values for both.
- Impossible request: say which requirement blocks it, how many recipes there would be without it \
and the closest available value; make clear that no requirement was dropped and ask which one to \
relax.
Return only the reply text."""

EXTRA_FIELDS = ("fat_g", "saturated_fat_g", "carbs_g", "sugar_g", "fiber_g", "sodium_mg", "cholesterol_mg")


def recipe_facts(recipe: dict) -> dict:
    nutrition = recipe.get("nutrition_per_serving", recipe)
    facts = {"recipe_id": recipe["recipe_id"], "title": recipe["title"],
             "duration_min": recipe["duration_min"]}
    for f in NUMERIC_FIELDS:
        if f != "duration_min" and f in nutrition:
            facts[f] = nutrition[f]
    return facts


def build_evidence(action: str, constraints, objectives, recipes=(), validations=None,
                   tradeoff=None, diagnosis=None, blocking=None, total_matches=None) -> dict:
    """Everything the explanation may say, as plain data."""
    checks = {v["recipe_id"]: v["checks"] for v in (validations or [])}
    ev = {
        "action": action,
        "requirements": [{"description": c.describe(), **c.to_dict()} for c in constraints],
        "objectives": [{"target": o.target, "direction": o.direction,
                        "meaning": f"{'lowest' if o.direction == 'min' else 'highest'} {o.target}"}
                       for o in objectives],
        "recipes": [],
    }
    for r in recipes:
        item = recipe_facts(r)
        if r["recipe_id"] in checks:
            item["checks"] = [{"requirement": c["description"], "satisfied": c["satisfied"],
                               "evidence": c["evidence"]} for c in checks[r["recipe_id"]]]
        ev["recipes"].append(item)
    if tradeoff:
        titles = {r["recipe_id"]: r["title"] for r in recipes}
        ev["tradeoff"] = [{"objective": f"{'lowest' if o['direction'] == 'min' else 'highest'} {o['target']}",
                           "field": o["target"], "recipe": titles.get(rid)} for o, rid in tradeoff
                          if rid in titles]
    if diagnosis is not None:
        ev["diagnosis"] = {
            "total_matches": diagnosis["total_matches"],
            "blocking_requirement": blocking["description"] if blocking else None,
            "matches_if_dropped": blocking["matches_if_dropped"] if blocking else None,
            "closest_available": blocking.get("closest_available") if blocking else None,
            "unit": NUMERIC_FIELDS.get(blocking["constraint"]["target"]) if blocking else None,
            "per_requirement": [{"requirement": e["description"], "matches_if_dropped": e["matches_if_dropped"],
                                 "closest_available": e.get("closest_available")}
                                for e in diagnosis["per_constraint"]],
        }
    if total_matches is not None:
        ev["total_matches"] = total_matches
    return ev


# ---------------------------------------------------------------------- the reviewer
_NUM = r"(\d+(?:[.,]\d+)?)"
# order matters: "saturated fat" before "fat", "mg" fields before bare "g"
_UNIT_PATTERNS = [
    ("saturated_fat_g", rf"{_NUM}\s*(?:g|grams?)\s*(?:of\s+)?saturated fat"),
    ("sodium_mg", rf"{_NUM}\s*(?:mg|milligrams?)\s*(?:of\s+)?sodium"),
    ("cholesterol_mg", rf"{_NUM}\s*(?:mg|milligrams?)\s*(?:of\s+)?cholesterol"),
    ("protein_g", rf"{_NUM}\s*(?:g|grams?)\s*(?:of\s+)?protein"),
    ("fat_g", rf"{_NUM}\s*(?:g|grams?)\s*(?:of\s+)?fat\b"),
    ("sugar_g", rf"{_NUM}\s*(?:g|grams?)\s*(?:of\s+)?sugars?"),
    ("fiber_g", rf"{_NUM}\s*(?:g|grams?)\s*(?:of\s+)?fib(?:er|re)"),
    ("carbs_g", rf"{_NUM}\s*(?:g|grams?)\s*(?:of\s+)?(?:carbs|carbohydrates?)"),
    ("calories_kcal", rf"{_NUM}\s*(?:kcal|calories?)\b"),
    ("duration_min", rf"{_NUM}\s*(?:min(?:ute)?s?)\b"),
    ("count", rf"{_NUM}\s*(?:recipes?|options?|matches|dishes)\b"),
]
_MEETS = re.compile(r"\b(meets?|satisf(?:y|ies)|fits?|match(?:es)?)\b[^.;]{0,40}\b(all|every|each)\b[^.;]{0,30}"
                    r"\b(requirements?|constraints?|criteria|wishes|needs)", re.I)
_NAME_WINDOW = 260  # characters: a number is attributed to the nearest recipe named within this span
_TYPOGRAPHY = str.maketrans({" ": " ", " ": " ", " ": " ", "‑": "-", "‐": "-",
                             "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"'})


def _normalize(text: str) -> str:
    """Models write '20 min', 'main‑dish' and curly quotes; titles in the data use plain
    characters. Compare both in the same plain form. Markdown emphasis is dropped too."""
    return text.translate(_TYPOGRAPHY).replace("**", "").replace("__", "").replace("\\", "")


def _close(claimed: float, actual: float) -> bool:
    return abs(claimed - actual) <= max(0.051 * abs(actual), 0.55) or round(claimed) == round(actual)


def _title_mentions(text: str, recipes: list[dict]) -> list[tuple[int, int, dict]]:
    """(start, end, recipe) for every mention of a recipe title (full, or the part before '(')."""
    low = text.lower()
    found = []
    for r in recipes:
        variants = {_normalize(r["title"]).lower()}
        short = _normalize(r["title"]).split(" (")[0].lower()
        if len(short) >= 6:
            variants.add(short)
        for v in variants:
            start = 0
            while True:
                i = low.find(v, start)
                if i < 0:
                    break
                found.append((i, i + len(v), r))
                start = i + len(v)
    found.sort()
    return found


def review(text: str, evidence: dict, db: RecipeDB) -> dict:
    """Check every quantity and 'meets all requirements' statement in `text` against the evidence.
    Returns {"claims": [...], "unsupported": [...], "n_checked": int, "no_recipe_named": bool}."""
    text = _normalize(text)
    recipes = evidence.get("recipes", [])
    mentions = _title_mentions(text, recipes)
    inside_title = [(s, e) for s, e, _ in mentions]
    req_values = {(r["target"], float(r["value"])) for r in evidence.get("requirements", [])
                  if r.get("value") is not None}
    diag = evidence.get("diagnosis") or {}
    diag_counts = {diag.get("total_matches"), diag.get("matches_if_dropped"), evidence.get("total_matches"),
                   len(recipes)} | {e.get("matches_if_dropped") for e in diag.get("per_requirement", [])}
    diag_values = {diag.get("closest_available")} | {e.get("closest_available") for e in diag.get("per_requirement", [])}
    diag_counts.discard(None); diag_values.discard(None)

    def recipe_before(pos: int):
        best = None
        for s, e, r in mentions:
            if e <= pos and pos - e <= _NAME_WINDOW:
                best = r
        return best

    claims, taken = [], set()
    for field, pat in _UNIT_PATTERNS:
        for m in re.finditer(pat, text, flags=re.I):
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue  # already claimed by a more specific pattern
            if any(s <= m.start() and m.end() <= e for s, e in inside_title):
                continue  # a number that is part of a recipe title ("..., 20 Minutes Max")
            taken.add((m.start(), m.end()))
            claimed = float(m.group(1).replace(",", "."))
            claim = {"text": m.group(0), "field": field, "claimed": claimed, "recipe": None,
                     "actual": None, "supported": False, "basis": None}
            if field == "count":
                claim["supported"] = any(_close(claimed, float(c)) for c in diag_counts)
                claim["basis"] = "count in evidence" if claim["supported"] else None
            else:
                r = recipe_before(m.start())
                if r is not None and field in r:
                    claim["recipe"], claim["actual"] = r["title"], r[field]
                    claim["supported"] = _close(claimed, float(r[field]))
                    claim["basis"] = "named recipe" if claim["supported"] else None
                if not claim["supported"] and (field, claimed) in req_values:
                    claim["supported"], claim["basis"] = True, "user requirement"
                if not claim["supported"] and any(_close(claimed, float(v)) for v in diag_values):
                    claim["supported"], claim["basis"] = True, "diagnosis"
                if not claim["supported"]:
                    # no recipe named just before, or the sentence refers back to another one
                    # ("the salad is best with 0.2 g fat"): accept a value that belongs to some
                    # listed recipe, and record which one so the attribution stays auditable
                    for cand in recipes:
                        if field in cand and _close(claimed, float(cand[field])):
                            claim["recipe"], claim["actual"] = cand["title"], cand[field]
                            claim["supported"] = True
                            claim["basis"] = "a listed recipe" if r is None else "another listed recipe"
                            break
            claims.append(claim)

    for m in _MEETS.finditer(text):
        r = recipe_before(m.start())
        if r is None:
            continue
        cs = [Constraint.from_dict({k: v for k, v in req.items() if k in ("kind", "target", "value")})
              for req in evidence.get("requirements", [])]
        ok = satisfies_all(db.recipes[r["recipe_id"]], cs) if cs else True
        claims.append({"text": m.group(0), "field": "meets_all", "claimed": True, "recipe": r["title"],
                       "actual": ok, "supported": bool(ok), "basis": "constraint checker" if ok else None})

    unsupported = [c for c in claims if not c["supported"]]
    return {"claims": claims, "unsupported": unsupported, "n_checked": len(claims),
            "no_recipe_named": bool(recipes) and not mentions}
