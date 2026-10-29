"""Optional extensions: reviewer, model-written explanations, preference ranking."""
import pytest

from src.agent import Agent
from src.explain import build_evidence, review
from src.extract_rules import RuleExtractor
from src.schema import Constraint
from src.tools import RecipeDB, preference_score, search_recipes


@pytest.fixture(scope="module")
def db():
    return RecipeDB.load()


@pytest.fixture(scope="module")
def rules():
    return RuleExtractor()


def _recipe(db, constraints):
    return db.recipes[search_recipes(db, constraints=constraints, k=1)["results"][0]["recipe_id"]]


def test_reviewer_accepts_correct_numbers_and_rejects_wrong_ones(db):
    cs = [Constraint("diet", "vegetarian"), Constraint("max", "duration_min", 25)]
    r = _recipe(db, cs)
    ev = build_evidence("recommend", cs, [], [r])
    good = (f"I recommend {r['title']} ({r['duration_min']:g} min, {r['calories_kcal']:g} kcal, "
            f"{r['protein_g']:g} g protein per serving). It meets all your requirements, including "
            f"the 25 min limit.")
    rep = review(good, ev, db)
    assert rep["n_checked"] >= 4 and rep["unsupported"] == [] and not rep["no_recipe_named"]
    bad = f"I recommend {r['title']} ({r['duration_min']:g} min, {r['protein_g'] + 20:g} g protein per serving)."
    rep = review(bad, ev, db)
    assert len(rep["unsupported"]) == 1 and rep["unsupported"][0]["field"] == "protein_g"


def test_reviewer_tolerates_rounding_and_flags_false_meets_statement(db):
    cs = [Constraint("max", "duration_min", 25)]
    r = _recipe(db, cs)
    ev = build_evidence("recommend", cs, [], [r])
    rounded = f"{r['title']} has about {round(r['calories_kcal'])} calories and takes {r['duration_min']:g} minutes."
    assert review(rounded, ev, db)["unsupported"] == []
    # a recipe that fails the requirement, claimed to meet it
    heavy_cs = [Constraint("max", "duration_min", 5)]
    heavy = _recipe(db, [Constraint("min", "duration_min", 120)])
    ev2 = build_evidence("recommend", heavy_cs, [], [heavy])
    rep = review(f"{heavy['title']} meets all your requirements.", ev2, db)
    assert any(c["field"] == "meets_all" and not c["supported"] for c in rep["unsupported"])


def test_reviewer_checks_infeasible_counts_and_flags_missing_recipe_name(db):
    cs = [Constraint("diet", "vegan"), Constraint("course", "breakfast"), Constraint("min", "protein_g", 25)]
    from src.tools import diagnose_constraints
    diag = diagnose_constraints(db, cs)
    blocking = max(diag["per_constraint"], key=lambda e: e["matches_if_dropped"])
    ev = build_evidence("report_infeasible", cs, [], diagnosis=diag, blocking=blocking)
    n = blocking["matches_if_dropped"]
    ok = review(f"Nothing matches. Without the protein requirement there would be {n} recipes.", ev, db)
    assert ok["unsupported"] == []
    wrong = review(f"Nothing matches. Without it there would be {n + 7} recipes.", ev, db)
    assert len(wrong["unsupported"]) == 1
    r = _recipe(db, [Constraint("diet", "vegetarian")])
    rep = review("Here is something nice for you.", build_evidence("recommend", [], [], [r]), db)
    assert rep["no_recipe_named"]


class FakeLLM:
    """Stands in for an LLMExtractor: returns pre-set explanation texts."""
    name = "fake"

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def complete(self, system, user, max_tokens=1200):
        self.calls += 1
        return self.texts.pop(0)


def test_llm_explanation_is_used_when_review_passes_and_replaced_when_it_fails(db, rules):
    cs = [Constraint("diet", "vegetarian"), Constraint("course", "main-dish")]
    r = _recipe(db, cs)  # the agent's first recommendation for this request
    good = f"Try {r['title']}: {r['duration_min']:g} min and {r['calories_kcal']:g} kcal per serving."
    wrong = f"Try {r['title']}: it has {r['protein_g'] + 50:g} g protein per serving."
    agent = Agent(db, rules, explain="llm", review=True, llm=FakeLLM([good, wrong]))
    assert agent.name == "agent_llmexp_rev"
    agent.reset_dialogue()
    t1 = agent.turn("Suggest a vegetarian dinner.")
    assert t1["message"] == good and t1["review"]["corrected"] is False
    agent.reset_dialogue()
    t2 = agent.turn("Suggest a vegetarian dinner.")
    assert t2["review"]["corrected"] is True and t2["message"] == t2["template_message"]
    assert t2["llm_message"] == wrong and len(t2["review"]["unsupported"]) == 1


def test_review_only_mode_checks_the_template_and_passes(db, rules):
    agent = Agent(db, rules, review=True)
    assert agent.name == "agent_rev"
    t = agent.turn("A chicken dinner under 40 minutes.")
    assert t["review"]["n_checked"] >= 3 and t["review"]["unsupported"] == []


def test_preference_ranking_keeps_the_feasible_set_but_changes_the_order(db):
    cs = [Constraint("diet", "vegetarian"), Constraint("course", "main-dish")]
    pop = search_recipes(db, constraints=cs, k=3)
    pref = search_recipes(db, constraints=cs, k=3, rank="preference")
    assert pop["total_matches"] == pref["total_matches"]
    top_pop = db.recipes[pop["results"][0]["recipe_id"]]
    top_pref = db.recipes[pref["results"][0]["recipe_id"]]
    assert preference_score(top_pref) >= preference_score(top_pop)
    assert 100 <= top_pref["calories_kcal"] <= 800  # the 1,727 kcal pizza dough is no longer first
    assert all(0.0 <= preference_score(db.recipes[r["recipe_id"]]) <= 1.0 for r in pref["results"])


def test_agent_with_preference_ranking_still_validates(db, rules):
    agent = Agent(db, rules, ranking="preference")
    assert agent.name == "agent_pref"
    t = agent.turn("Suggest a vegetarian dinner.")
    assert t["action"] == "recommend" and all(r["verified"] for r in t["recommendations"])
    assert any(c["tool"] == "search_recipes" and c["args"].get("rank") == "preference" for c in t["tool_calls"])
