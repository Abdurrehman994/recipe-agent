"""Agent decision policy (spec item 6) with grounded, templated explanations (spec item 7).

    agent = Agent(RecipeDB.load(), RuleExtractor())
    record = agent.turn("Suggest a vegetarian dinner.")     # -> log record (dict)

Per user turn the agent
  1. extracts constraint updates (rule or LLM extractor) and updates the dialogue state,
  2. chooses an action: clarify | recommend (search, then inspect each candidate) |
     report_infeasible,
  3. calls the deterministic tools for every factual check and never relaxes a constraint by
     itself: when nothing matches it explains which requirement blocks the request and asks,
  4. writes a message whose factual statements all come from tool output, together with a
     structured list of those `claims` so they can be verified against the database
     (`verify_claims`).

User turns are scripted by the scenarios, so runs are reproducible.

System configurations (the ablation study, spec item 8):
  mode="rag"         conversational RAG baseline: BM25 over the transcript so far, no
                     structured state, no filtering, no validation; always recommends.
  mode="rag_memory"  RAG + structured memory: the dialogue state filters retrieval, but there
                     is no policy: no clarification, no validation, and when nothing matches it
                     falls back to unfiltered retrieval (a silent relaxation, logged as such).
  mode="agent"       the full policy above.
    memory=False     the state is rebuilt from the current turn only (memory ablation).
    validate=False   constraints reach retrieval only as soft query words, with no hard filter
                     and no validate_constraints check; infeasibility cannot be detected
                     (validation ablation).
"""
from __future__ import annotations

from src.extract_rules import RuleExtractor
from src.schema import NUMERIC_FIELDS, Constraint, Objective
from src.state import DialogueState
from src.tools import (RecipeDB, check, diagnose_constraints, get_recipe_details, search_recipes,
                       validate_constraints)

MODES = ("rag", "rag_memory", "agent")
CLARIFY_QUESTION = ("Happy to help. To narrow it down: what kind of meal are you after "
                    "(breakfast, lunch, dinner, dessert, ...), do you have any dietary restrictions "
                    "or ingredients to avoid, and how much time do you have?")
FACT_FIELDS = ("duration_min", "calories_kcal", "protein_g")  # quoted for every recommendation
# Dish types are not constraints (scenarios/README.md) but they are a retrieval topic: "find me
# a curry" searches for curries, and the topic persists until the user resets the request.
DISH_WORDS = {
    "curry", "curries", "stir-fry", "stir fry", "stirfry", "tacos", "taco", "burrito", "pizza",
    "burger", "burgers", "casserole", "sandwich", "sandwiches", "wrap", "wraps", "smoothie",
    "pie", "cake", "cookies", "cookie", "brownies", "muffins", "pancakes", "waffles", "risotto",
    "lasagna", "lasagne", "chili", "omelette", "omelet", "quiche", "pilaf", "paella", "stew",
    "chowder", "kebab", "kebabs", "meatballs", "dumplings", "fritters", "pudding", "crumble",
    "cheesecake", "pasta bake", "enchiladas", "fajitas", "quesadilla", "frittata",
}


def topic_words(utterance: str) -> list[str]:
    text = utterance.lower()
    return [w for w in sorted(DISH_WORDS, key=len, reverse=True) if w in text]


def _field_name(field: str) -> str:
    return "time" if field == "duration_min" else field.rsplit("_", 1)[0].replace("_", " ")


def _facts(recipe: dict) -> str:
    """'25 min, 420 kcal, 31 g protein per serving' from a summary or details dict."""
    nutrition = recipe.get("nutrition_per_serving", recipe)
    return (f"{recipe['duration_min']:g} min, {nutrition['calories_kcal']:g} kcal, "
            f"{nutrition['protein_g']:g} g protein per serving")


def _value(recipe: dict, field: str):
    return recipe["duration_min"] if field == "duration_min" else \
        recipe.get("nutrition_per_serving", recipe)[field]


class Agent:
    def __init__(self, db: RecipeDB, extractor=None, mode: str = "agent", memory: bool = True,
                 validate: bool = True, k: int = 3):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        self.db = db
        self.extractor = extractor or RuleExtractor()
        self.mode, self.memory, self.validate, self.k = mode, memory, validate, k
        self.reset_dialogue()

    @property
    def name(self) -> str:
        """Configuration name used in log and result file names."""
        if self.mode != "agent":
            return self.mode
        return "agent" + ("" if self.memory else "_nomem") + ("" if self.validate else "_noval")

    def reset_dialogue(self) -> None:
        self.state = DialogueState()
        self.transcript: list[str] = []
        self.topic: list[str] = []  # dish words for retrieval, e.g. ["curry"]
        self.turn_no = 0

    # ------------------------------------------------------------------ tool wrapper
    def _tool(self, log: dict, name: str, **kwargs):
        fn = {"search_recipes": search_recipes, "get_recipe_details": get_recipe_details,
              "validate_constraints": validate_constraints,
              "diagnose_constraints": diagnose_constraints}[name]
        result = fn(self.db, **kwargs)
        args = {k: ([c.to_dict() for c in v] if k in ("constraints", "objectives") else v)
                for k, v in kwargs.items()}
        if name == "search_recipes":
            summary = {"total_matches": result["total_matches"],
                       "results": [(r["recipe_id"], r["title"]) for r in result["results"]]}
        elif name == "get_recipe_details":
            summary = {"recipe_id": result["recipe_id"], "title": result["title"]}
        elif name == "validate_constraints":
            summary = {"recipe_id": result["recipe_id"], "all_satisfied": result["all_satisfied"],
                       "failed": [c["description"] for c in result["checks"] if not c["satisfied"]]}
        else:
            summary = result
        log["tool_calls"].append({"tool": name, "args": args, "result": summary})
        return result

    # ------------------------------------------------------------------ one turn
    def turn(self, utterance: str) -> dict:
        self.turn_no += 1
        self.transcript.append(utterance)
        log = {"turn": self.turn_no, "user": utterance, "config": self.name,
               "extractor": getattr(self.extractor, "name", None), "extraction": None,
               "state": [], "objectives": [], "action": None, "tool_calls": [],
               "recommendations": [], "rejected": [], "relaxed": False, "claims": [],
               "message": ""}
        if self.mode == "rag":
            return self._rag_turn(log)
        ext = self.extractor.extract(utterance, self.state)
        log["extraction"] = ext.to_dict()
        if not self.memory or ext.reset:
            self.topic = []
        if not self.memory:
            self.state = DialogueState()
        self.state.apply(ext, self.turn_no)
        self.topic += [w for w in topic_words(utterance) if w not in self.topic]
        log["topic"] = list(self.topic)
        log["state"] = self.state.to_active()
        log["objectives"] = [o.to_dict() for o in self.state.objectives]
        if self.mode == "rag_memory":
            return self._rag_memory_turn(log, utterance)
        return self._agent_turn(log, utterance)

    # ------------------------------------------------------------------ baselines
    def _rag_turn(self, log: dict) -> dict:
        res = self._tool(log, "search_recipes", query=" ".join(self.transcript), k=self.k)
        return self._recommend(log, res["results"], [], [], verified=False)

    def _rag_memory_turn(self, log: dict, utterance: str) -> dict:
        constraints, objectives = self.state.constraints, self.state.objectives
        res = self._tool(log, "search_recipes", query=utterance, constraints=constraints,
                         objectives=objectives, k=self.k)
        if res["total_matches"] == 0:  # no policy: fall back to unfiltered retrieval
            log["relaxed"] = True
            res = self._tool(log, "search_recipes", query=" ".join(self.transcript), k=self.k)
        return self._recommend(log, res["results"], constraints, objectives, verified=False)

    # ------------------------------------------------------------------ full policy
    def _agent_turn(self, log: dict, utterance: str) -> dict:
        constraints, objectives = self.state.constraints, self.state.objectives
        query = " ".join(self.topic)
        if not constraints:
            if not query:  # nothing to search or filter with
                return self._clarify(log, objectives)
            res = self._tool(log, "search_recipes", query=query, objectives=objectives, k=self.k)
            return self._recommend(log, res["results"], [], objectives, verified=False)
        if not self.validate:  # ablation: constraints as soft query words only
            words = " ".join(c.target.replace("_", " ").replace("-", " ") for c in constraints)
            res = self._tool(log, "search_recipes", query=f"{utterance} {words}",
                             objectives=objectives, k=self.k)
            return self._recommend(log, res["results"], constraints, objectives, verified=False)

        res = self._tool(log, "search_recipes", query=query, constraints=constraints,
                         objectives=objectives, k=self.k)
        if res["total_matches"] == 0:
            return self._report_infeasible(log, constraints)

        candidates = res["results"]
        if len(objectives) >= 2:  # competing objectives: best candidate per objective
            picks = []
            for o in objectives:
                top = self._tool(log, "search_recipes", query=query, constraints=constraints,
                                 objectives=[o], k=1)["results"]
                picks += [(o, r) for r in top]
            seen, candidates = set(), []
            for _, r in picks:
                if r["recipe_id"] not in seen:
                    seen.add(r["recipe_id"])
                    candidates.append(r)
            log["tradeoff"] = [(o.to_dict(), r["recipe_id"]) for o, r in picks]

        verified = []
        for r in candidates:  # inspect each candidate before recommending it
            details = self._tool(log, "get_recipe_details", recipe_id=r["recipe_id"])
            val = self._tool(log, "validate_constraints", recipe_id=r["recipe_id"],
                             constraints=constraints)
            if val["all_satisfied"]:
                verified.append((details, val))
            else:
                log["rejected"].append({"recipe_id": r["recipe_id"], "title": r["title"],
                                        "failed": [c["description"] for c in val["checks"]
                                                   if not c["satisfied"]]})
        if not verified:
            return self._report_infeasible(log, constraints)
        return self._recommend(log, [d for d, _ in verified], constraints, objectives,
                               verified=True, validations=[v for _, v in verified])

    # ------------------------------------------------------------------ actions
    def _clarify(self, log: dict, objectives: list[Objective]) -> dict:
        log["action"] = "clarify"
        prefix = ""
        if objectives:
            wants = " and ".join(f"{'low' if o.direction == 'min' else 'high'} {_field_name(o.target)}"
                                 for o in objectives)
            prefix = f"Noted that you want {wants}. "
        log["message"] = prefix + CLARIFY_QUESTION
        return log

    def _report_infeasible(self, log: dict, constraints: list[Constraint]) -> dict:
        log["action"] = "report_infeasible"
        diag = self._tool(log, "diagnose_constraints", constraints=constraints)
        wanted = "; ".join(c.describe() for c in constraints)
        blocking = max(diag["per_constraint"], key=lambda e: e["matches_if_dropped"])
        n = blocking["matches_if_dropped"]
        detail = (f"Without the requirement \"{blocking['description']}\" there would be {n} "
                  f"recipe{'s' if n != 1 else ''}")
        if "closest_available" in blocking:
            unit = NUMERIC_FIELDS[blocking["constraint"]["target"]]
            detail += f", and the closest available value is {blocking['closest_available']:g} {unit}"
        detail += "."
        log["message"] = (f"I couldn't find any recipe in the database that is {wanted}. {detail} "
                          f"I haven't dropped any of your requirements; tell me which one you "
                          f"would like to relax.")
        log["claims"] = [{"type": "no_match", "constraints": [c.to_dict() for c in constraints]},
                         {"type": "matches_if_dropped", "constraint": blocking["constraint"],
                          "others": [c.to_dict() for c in constraints
                                     if c.to_dict() != blocking["constraint"]], "value": n}]
        return log

    def _recommend(self, log: dict, recipes: list[dict], constraints: list[Constraint],
                   objectives: list[Objective], verified: bool, validations=None) -> dict:
        log["action"] = "recommend"
        log["recommendations"] = [{"recipe_id": r["recipe_id"], "title": r["title"],
                                   "verified": verified} for r in recipes]
        if not recipes:
            log["message"] = "I couldn't find anything to recommend."
            return log
        for r in recipes:
            for f in FACT_FIELDS:
                log["claims"].append({"type": "value", "recipe_id": r["recipe_id"], "field": f,
                                      "value": _value(r, f)})
        if not verified:  # baselines / topic-only search: facts quoted, nothing claimed
            lines = [f"{r['title']} ({_facts(r)})" for r in recipes]
            topic = f" for {' '.join(log.get('topic') or [])}" if log.get("topic") and not constraints else ""
            log["message"] = f"Here are some options{topic}: " + "; ".join(lines) + "."
            return log

        met = "; ".join(c.describe() for c in constraints)
        for v in validations:
            for ch in v["checks"]:
                log["claims"].append({"type": "constraint", "recipe_id": v["recipe_id"],
                                      "constraint": ch["constraint"], "satisfied": True})
        if len(objectives) >= 2 and len(recipes) >= 2 and "tradeoff" in log:
            by_id = {r["recipe_id"]: r for r in recipes}
            parts = []
            for o_dict, rid in log["tradeoff"]:
                if rid not in by_id:
                    continue
                r = by_id[rid]
                label = ("quickest" if o_dict["target"] == "duration_min" else
                         f"{'lowest' if o_dict['direction'] == 'min' else 'highest'} "
                         f"{_field_name(o_dict['target'])}")
                parts.append(f"the {label} option is {r['title']} ({_facts(r)})")
            log["message"] = ("These objectives pull in different directions, so here is the "
                              "trade-off: " + "; ".join(parts) + f". Both meet all your "
                              f"requirements ({met}).")
            return log
        if len(objectives) >= 2 and len(recipes) == 1:
            r = recipes[0]
            log["message"] = (f"{r['title']} ({_facts(r)}) is the best option on every count: "
                              f"it meets all your requirements ({met}).")
            return log

        ranked = ""
        if objectives:
            ranked = " Ranked by " + " and ".join(
                f"{'lowest' if o.direction == 'min' else 'highest'} {_field_name(o.target)}"
                for o in objectives) + "."
        first, rest = recipes[0], recipes[1:]
        evidence = "; ".join(ch["evidence"] for ch in validations[0]["checks"])
        msg = (f"I recommend {first['title']} ({_facts(first)}). It meets all your requirements "
               f"({met}): {evidence}.")
        if rest:
            msg += " Also verified: " + "; ".join(f"{r['title']} ({_facts(r)})" for r in rest) + "."
        log["message"] = msg + ranked
        return log


# ---------------------------------------------------------------------- claim checking
def verify_claims(db: RecipeDB, claims: list[dict]) -> list[dict]:
    """Re-check every factual claim in a log record against the database (deterministic).
    Returns the claims that are NOT supported."""
    unsupported = []
    for cl in claims:
        if cl["type"] == "value":
            r = db.recipes[cl["recipe_id"]]
            ok = abs(float(r[cl["field"]]) - float(cl["value"])) < 1e-6
        elif cl["type"] == "constraint":
            r = db.recipes[cl["recipe_id"]]
            ok = check(r, Constraint.from_dict(cl["constraint"]))[0] == cl["satisfied"]
        elif cl["type"] == "no_match":
            cs = [Constraint.from_dict(c) for c in cl["constraints"]]
            ok = search_recipes(db, constraints=cs, k=1)["total_matches"] == 0
        elif cl["type"] == "matches_if_dropped":
            cs = [Constraint.from_dict(c) for c in cl["others"]]
            ok = search_recipes(db, constraints=cs, k=1)["total_matches"] == cl["value"]
        else:
            ok = False
        if not ok:
            unsupported.append(cl)
    return unsupported
