"""Retrieval and validation tools. All checks are deterministic code, never LLM judgment.

    db = RecipeDB.load()
    search_recipes(db, query="pasta", constraints=[...], objectives=[...], k=5)
    get_recipe_details(db, recipe_id)
    validate_constraints(db, recipe_id, constraints)
    diagnose_constraints(db, constraints)      # why is nothing matching? what is closest?

Constraints/objectives may be passed as schema objects or as plain dicts (LLM tool calls).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from src.data_prep import CATEGORIES, ingredient_categories, strip_notes

line_categories = lru_cache(maxsize=None)(ingredient_categories)
clean_line = lru_cache(maxsize=None)(strip_notes)
from src.schema import NUMERIC_FIELDS, Constraint, Objective

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "recipes.jsonl"
DIET_FLAGS = ["vegetarian", "pescatarian", "vegan", "gluten_free", "dairy_free", "nut_free", "egg_free"]
CATEGORY_SYNONYMS = {  # user words -> keyword category from data_prep.CATEGORIES
    "mushrooms": "mushroom", "nut": "nuts", "tree nuts": "nuts", "eggs": "egg",
    "seafood": "shellfish", "red meat": "meat", "milk products": "dairy",
}
_TOKEN = re.compile(r"[a-z]+")


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if len(t) > 1]


def stem(word: str) -> str:
    """Tiny plural stemmer: tomatoes -> tomato, berries -> berry, onions -> onion."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("oes"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


# ---------------------------------------------------------------- database + BM25
class RecipeDB:
    def __init__(self, recipes: list[dict]):
        self.recipes = {r["recipe_id"]: r for r in recipes}
        self.ids = list(self.recipes)
        self._build_index()

    @classmethod
    def load(cls, path: Path = DB_PATH) -> RecipeDB:
        with open(path, encoding="utf-8") as f:
            return cls([json.loads(line) for line in f])

    def __len__(self):
        return len(self.recipes)

    def get(self, recipe_id) -> dict:
        try:
            return self.recipes[int(recipe_id)]
        except (KeyError, ValueError, TypeError):
            raise KeyError(f"unknown recipe_id: {recipe_id}") from None

    def _build_index(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.doc_len = {}
        for rid, r in self.recipes.items():
            text = " ".join([r["title"]] * 2 + r["ingredients"] + r["tags"] + r["courses"]
                            + [r["description"]])
            counts = Counter(tokenize(text))
            self.doc_len[rid] = sum(counts.values())
            for tok, c in counts.items():
                self.postings[tok][rid] = c
        self.avg_len = sum(self.doc_len.values()) / len(self.doc_len)

    def bm25(self, query: str, candidates=None) -> dict[int, float]:
        n = len(self.recipes)
        scores: dict[int, float] = defaultdict(float)
        for tok in set(tokenize(query)):
            post = self.postings.get(tok)
            if not post:
                continue
            idf = math.log(1 + (n - len(post) + 0.5) / (len(post) + 0.5))
            for rid, tf in post.items():
                if candidates is not None and rid not in candidates:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[rid] / self.avg_len)
                scores[rid] += idf * tf * (self.k1 + 1) / norm
        return scores


# ---------------------------------------------------------------- constraint checking
def as_constraint(c) -> Constraint:
    return c if isinstance(c, Constraint) else Constraint.from_dict(c)


def as_objective(o) -> Objective:
    return o if isinstance(o, Objective) else Objective(**o)


@lru_cache(maxsize=4096)
def _term_pattern(term: str) -> re.Pattern:
    words = [stem(w) for w in term.lower().split()]
    body = r"\s+".join(re.escape(w) + r"(s|es)?" if not w.endswith("y") else
                       re.escape(w[:-1]) + r"(y|ies)" for w in words)
    return re.compile(rf"\b{body}\b")


_FLAVORING = re.compile(r"^\s*(broth|stock|bouillon|soup|base|gravy|seasoning|flavou?red|"
                        r"flavou?ring|fat|drippings|granules|cubes?)\b")


def find_ingredient(recipe: dict, term: str, strict: bool = True) -> list[str]:
    """Return the recipe's ingredient lines that contain `term` (an ingredient or category).

    strict=True (used for exclusions): any mention counts, e.g. chicken broth contains chicken.
    strict=False (used for inclusions): a mention that is only a flavoring does not count, so
    "a chicken dinner" is not satisfied by a recipe whose only chicken is the broth."""
    term = term.lower().strip()
    category = CATEGORY_SYNONYMS.get(term, term)
    lines = recipe["ingredients_with_quantities"]
    if category in CATEGORIES:
        if category not in recipe["ingredient_categories"]:  # precomputed fast path
            return []
        return [line for line in lines if category in line_categories(line)]
    pattern = _term_pattern(term)
    hits = []
    for line in lines:
        text = clean_line(line)
        matches = list(pattern.finditer(text))
        if strict:
            if matches:
                hits.append(line)
        elif any(not _FLAVORING.match(text[m.end():]) for m in matches):
            hits.append(line)
    return hits


def check(recipe: dict, c: Constraint) -> tuple[bool, str]:
    """Check one constraint; returns (satisfied, human-readable evidence from the data)."""
    if c.kind == "diet":
        ok = bool(recipe[c.target])
        if ok:
            return True, f"{c.target.replace('_', '-')}: no conflicting ingredients"
        bad = {"vegetarian": {"meat", "poultry", "fish", "shellfish"},
               "pescatarian": {"meat", "poultry"},
               "vegan": {"meat", "poultry", "fish", "shellfish", "dairy", "egg", "honey"},
               "gluten_free": {"gluten"}, "dairy_free": {"dairy"}, "nut_free": {"nuts"},
               "egg_free": {"egg"}}[c.target]
        lines = [l for l in recipe["ingredients_with_quantities"] if line_categories(l) & bad]
        return False, "conflicting ingredients: " + "; ".join(lines[:3])
    if c.kind == "course":
        ok = recipe["is_main"] if c.target == "main-dish" else c.target in recipe["courses"]
        return ok, f"courses: {', '.join(recipe['courses']) or 'none tagged'}"
    if c.kind in ("include", "exclude"):
        lines = find_ingredient(recipe, c.target, strict=c.kind == "exclude")
        if c.kind == "include":
            return (True, "contains: " + "; ".join(lines[:3])) if lines else \
                   (False, f"no ingredient matching '{c.target}'")
        return (False, "contains: " + "; ".join(lines[:3])) if lines else \
               (True, f"no ingredient matching '{c.target}'")
    value = recipe[c.target]
    unit = NUMERIC_FIELDS[c.target]
    ok = value <= c.value if c.kind == "max" else value >= c.value
    sign = ("<=" if ok else ">") if c.kind == "max" else (">=" if ok else "<")
    per = "" if c.target == "duration_min" else " per serving"
    return ok, f"{c.target} = {value:g} {unit}{per} ({sign} {c.value:g})"


def satisfies_all(recipe: dict, constraints: list[Constraint]) -> bool:
    return all(check(recipe, c)[0] for c in constraints)


def summarize(recipe: dict) -> dict:
    return {
        "recipe_id": recipe["recipe_id"], "title": recipe["title"],
        "duration_min": recipe["duration_min"], "calories_kcal": recipe["calories_kcal"],
        "protein_g": recipe["protein_g"], "courses": recipe["courses"],
        "diets": [d for d in DIET_FLAGS if recipe[d]],
        "rating": recipe["average_rating"], "n_ratings": recipe["number_of_ratings"],
    }


# ---------------------------------------------------------------- tools
REFERENCE_KCAL = 450.0  # a typical single-serving main course; the preference prior is centred here


def preference_score(r: dict) -> float:
    """Optional extension (preference-based ranking): 0..1, higher is better.
    Community rating, damped for few ratings, plus a portion-sanity prior that favours servings
    near a normal size and preparation times that are not extreme. Only the ORDER within the set of
    recipes that already satisfy every constraint is affected."""
    rating = (r["average_rating"] / 5.0) * min(1.0, math.log1p(r["number_of_ratings"]) / math.log1p(50))
    kcal = max(float(r["calories_kcal"]), 1.0)
    portion = 1.0 - min(1.0, abs(math.log(kcal / REFERENCE_KCAL)) / math.log(3.0))
    time_ok = 1.0 - min(1.0, max(0.0, float(r["duration_min"]) - 60.0) / 240.0)
    return 0.5 * rating + 0.3 * portion + 0.2 * time_ok


def search_recipes(db: RecipeDB, query: str = "", constraints=(), objectives=(), k: int = 5,
                   exclude_ids=(), rank: str = "popularity") -> dict:
    """Hard-filter by constraints, then rank by objectives (if any), BM25 relevance, and either
    popularity (default) or the preference score (rank="preference")."""
    constraints = [as_constraint(c) for c in constraints]
    objectives = [as_objective(o) for o in objectives]
    excluded = {int(i) for i in exclude_ids}
    pool = [rid for rid in db.ids if rid not in excluded
            and satisfies_all(db.recipes[rid], constraints)]
    rel = db.bm25(query, set(pool)) if query.strip() else {}

    def popularity(r):
        return r["average_rating"] * math.log1p(r["number_of_ratings"])

    tiebreak = preference_score if rank == "preference" else popularity

    def key(rid):
        r = db.recipes[rid]
        obj = tuple(r[o.target] if o.direction == "min" else -r[o.target] for o in objectives)
        return (*obj, -rel.get(rid, 0.0), -tiebreak(r))

    if query.strip() and not objectives:  # with a query and no objective, drop irrelevant hits
        relevant = [rid for rid in pool if rel.get(rid, 0) > 0]
        pool = relevant or pool
    ranked = sorted(pool, key=key)
    return {"total_matches": len(pool), "results": [summarize(db.recipes[r]) for r in ranked[:k]]}


def get_recipe_details(db: RecipeDB, recipe_id) -> dict:
    r = db.get(recipe_id)
    return {
        "recipe_id": r["recipe_id"], "title": r["title"], "description": r["description"],
        "duration_min": r["duration_min"], "servings": r["servings"],
        "serving_size_g": r["serving_size_g"],
        "nutrition_per_serving": {f: r[f] for f in NUMERIC_FIELDS if f != "duration_min"},
        "ingredients": r["ingredients_with_quantities"],
        "ingredient_categories": r["ingredient_categories"],
        "diets": [d for d in DIET_FLAGS if r[d]], "courses": r["courses"],
        "directions": r["directions"], "rating": r["average_rating"],
        "n_ratings": r["number_of_ratings"], "url": r["recipe_url"],
    }


def validate_constraints(db: RecipeDB, recipe_id, constraints) -> dict:
    r = db.get(recipe_id)
    checks = []
    for c in map(as_constraint, constraints):
        ok, evidence = check(r, c)
        checks.append({"constraint": c.to_dict(), "description": c.describe(),
                       "satisfied": ok, "evidence": evidence})
    return {"recipe_id": r["recipe_id"], "title": r["title"],
            "all_satisfied": all(ch["satisfied"] for ch in checks), "checks": checks}


def diagnose_constraints(db: RecipeDB, constraints) -> dict:
    """For an over-constrained request: how many matches appear if each constraint is dropped,
    and for numeric limits, the closest value available under the remaining constraints.
    Used to EXPLAIN infeasibility; the agent must not apply a relaxation on its own."""
    constraints = [as_constraint(c) for c in constraints]
    passes = {rid: [check(db.recipes[rid], c)[0] for c in constraints] for rid in db.ids}
    total = sum(all(p) for p in passes.values())
    report = []
    for i, c in enumerate(constraints):
        others_ok = [rid for rid, p in passes.items() if all(p[:i] + p[i + 1:])]
        entry = {"constraint": c.to_dict(), "description": c.describe(),
                 "matches_if_dropped": len(others_ok)}
        if c.kind in ("max", "min") and others_ok:
            values = [db.recipes[rid][c.target] for rid in others_ok]
            entry["closest_available"] = min(values) if c.kind == "max" else max(values)
        report.append(entry)
    return {"total_matches": total, "per_constraint": report}
