"""Agent decision policy: actions, no silent relaxation, grounded claims, ablation switches."""
import pytest

from src.agent import Agent, verify_claims
from src.extract_rules import RuleExtractor
from src.schema import Constraint, ConstraintUpdate, Objective, TurnExtraction
from src.tools import RecipeDB, satisfies_all


@pytest.fixture(scope="module")
def db():
    return RecipeDB.load()


@pytest.fixture(scope="module")
def rules():
    return RuleExtractor()


def dialogue(agent, *utterances):
    agent.reset_dialogue()
    return [agent.turn(u) for u in utterances]


def test_recommendations_satisfy_state_and_are_verified(db, rules):
    agent = Agent(db, rules)
    t1, t2, t3 = dialogue(agent, "Suggest a vegetarian dinner.", "Under 25 minutes.", "Avoid mushrooms.")
    assert [t["action"] for t in (t1, t2, t3)] == ["recommend"] * 3
    cs = [Constraint("diet", "vegetarian"), Constraint("max", "duration_min", 25),
          Constraint("exclude", "mushroom")]
    assert t3["recommendations"] and all(r["verified"] for r in t3["recommendations"])
    assert all(satisfies_all(db.recipes[r["recipe_id"]], cs) for r in t3["recommendations"])
    tools = [c["tool"] for c in t3["tool_calls"]]
    assert "search_recipes" in tools and "validate_constraints" in tools
    assert "meets all your requirements" in t3["message"]
    assert verify_claims(db, t3["claims"]) == []


def test_clarify_when_nothing_can_be_constrained(db, rules):
    agent = Agent(db, rules)
    (t1,) = dialogue(agent, "Suggest something healthy.")
    assert t1["action"] == "clarify" and t1["tool_calls"] == [] and t1["recommendations"] == []
    assert t1["message"].endswith("?")


def test_infeasible_is_reported_with_diagnosis_not_relaxed(db, rules):
    agent = Agent(db, rules)
    t1, t2 = dialogue(agent, "A vegan gluten-free dessert.",
                      "Under 5 minutes with at least 40 g of protein.")
    assert t2["action"] == "report_infeasible" and t2["recommendations"] == []
    assert any(c["tool"] == "diagnose_constraints" for c in t2["tool_calls"])
    assert "haven't dropped any of your requirements" in t2["message"]
    assert verify_claims(db, t2["claims"]) == []
    # the state still holds every constraint: nothing was relaxed
    kinds = {(a["kind"], a["target"]) for a in t2["state"]}
    assert ("max", "duration_min") in kinds and ("min", "protein_g") in kinds


def test_tradeoff_presents_one_option_per_objective(db, rules):
    agent = Agent(db, rules)
    *_, t = dialogue(agent, "A main dish under 350 calories.", "Which is faster and which has more protein?")
    assert t["action"] == "recommend" and "tradeoff" in t
    assert len(t["recommendations"]) >= 1
    if len(t["recommendations"]) >= 2:
        assert "quickest option" in t["message"] and "highest protein option" in t["message"]
    assert verify_claims(db, t["claims"]) == []


class FixedExtractor:
    """Returns pre-set extractions, so policy tests do not depend on parser coverage."""
    name = "fixed"

    def __init__(self, *extractions):
        self.extractions = iter(extractions)

    def extract(self, utterance, state=None):
        return next(self.extractions)


def test_tradeoff_states_the_compared_attribute(db):
    shrimp_dinner = TurnExtraction([ConstraintUpdate("add", Constraint("include", "shrimp")),
                                    ConstraintUpdate("add", Constraint("course", "main-dish")),
                                    ConstraintUpdate("add", Constraint("max", "calories_kcal", 500))])
    fastest_or_leanest = TurnExtraction(objectives=[Objective("duration_min", "min"),
                                                    Objective("fat_g", "min")])
    agent = Agent(db, FixedExtractor(shrimp_dinner, fastest_or_leanest))
    *_, t = dialogue(agent, "Shrimp for dinner, under 500 calories.",
                     "Which one is the fastest, and which one is lowest in fat?")
    assert t["action"] == "recommend" and "tradeoff" in t
    assert len(t["recommendations"]) == 2
    assert "quickest option" in t["message"] and "lowest fat option" in t["message"]
    for r in t["recommendations"]:  # every option shows its fat value, backed by a claim
        fat = db.recipes[r["recipe_id"]]["fat_g"]
        assert f"{fat:g} g fat" in t["message"]
        assert {"type": "value", "recipe_id": r["recipe_id"], "field": "fat_g",
                "value": fat} in t["claims"]
    assert verify_claims(db, t["claims"]) == []


def test_memory_ablation_forgets_earlier_turns(db, rules):
    agent = Agent(db, rules, memory=False)
    _, t2 = dialogue(agent, "Suggest a vegetarian dinner.", "Under 25 minutes.")
    assert {(a["kind"], a["target"]) for a in t2["state"]} == {("max", "duration_min")}
    assert agent.name == "agent_nomem"


def test_rag_memory_relaxes_silently_when_infeasible(db, rules):
    agent = Agent(db, rules, mode="rag_memory")
    _, t2 = dialogue(agent, "A vegan gluten-free dessert.",
                     "Under 5 minutes with at least 40 g of protein.")
    assert t2["action"] == "recommend" and t2["relaxed"] and t2["recommendations"]
    assert verify_claims(db, t2["claims"]) == []  # quoted facts are still true


def test_rag_baseline_has_no_state_and_always_recommends(db, rules):
    agent = Agent(db, rules, mode="rag")
    t1, t2 = dialogue(agent, "Suggest something healthy.", "Under 25 minutes.")
    assert t1["action"] == t2["action"] == "recommend"
    assert t1["extraction"] is None and t2["state"] == []
    assert all(c["tool"] == "search_recipes" for c in t2["tool_calls"])


def test_validation_ablation_never_reports_infeasible(db, rules):
    agent = Agent(db, rules, validate=False)
    _, t2 = dialogue(agent, "A vegan gluten-free dessert.",
                     "Under 5 minutes with at least 40 g of protein.")
    assert t2["action"] == "recommend"
    assert not any(c["tool"] == "validate_constraints" for c in t2["tool_calls"])
    assert agent.name == "agent_noval"


def test_dish_word_is_a_retrieval_topic_not_a_clarification(db, rules):
    agent = Agent(db, rules)
    t1, t2, t3 = dialogue(agent, "Can you find me a curry?", "By the way, I'm vegetarian.",
                          "Forget that, what's a good dessert?")
    assert t1["action"] == "recommend" and t1["topic"] == ["curry"] and t1["state"] == []
    assert all("curry" in r["title"].lower() for r in t1["recommendations"])
    assert t2["action"] == "recommend" and t2["topic"] == ["curry"]
    assert all(satisfies_all(db.recipes[r["recipe_id"]], [Constraint("diet", "vegetarian")])
               for r in t2["recommendations"])
    assert t3["topic"] == [] and t3["action"] == "recommend"  # reset clears the topic


def test_verify_claims_catches_a_false_fact(db):
    rid = next(iter(db.recipes))
    bad = [{"type": "value", "recipe_id": rid, "field": "protein_g", "value": -1}]
    assert verify_claims(db, bad) == bad
