import pytest

from src.extract_rules import RuleExtractor
from src.schema import Objective
from src.state import DialogueState


@pytest.fixture(scope="module")
def ex():
    return RuleExtractor()


def updates(ext):
    return {(u.op, u.constraint.describe(), u.persistent) for u in ext.updates}


@pytest.mark.parametrize("utterance, expected", [
    ("Suggest a vegetarian dinner.", {("add", "vegetarian", False), ("add", "course: main-dish", False)}),
    ("It should take under 25 minutes.", {("add", "time at most 25 min", False)}),
    ("Pasta in under half an hour", {("add", "time at most 30 min", False), ("add", "contains pasta", False)}),
    ("I'm allergic to nuts. Can you suggest a dessert?",
     {("add", "nut-free", True), ("add", "course: desserts", False)}),
    ("I mean under 400 calories and less than 500 mg of sodium.",
     {("add", "calories at most 400 kcal", False), ("add", "sodium at most 500 mg", False)}),
    ("protein over 25g and fat under 15 g",
     {("add", "protein at least 25 g", False), ("add", "fat at most 15 g", False)}),
    ("Something without onions, garlic or cilantro",
     {("add", "no onion", False), ("add", "no garlic", False), ("add", "no cilantro", False)}),
    ("I don't eat pork or beef.", {("add", "no pork", True), ("add", "no beef", True)}),
    ("No dairy, please.", {("add", "dairy-free", False)}),
    ("I'd like a chicken dinner.", {("add", "contains chicken", False), ("add", "course: main-dish", False)}),
    ("Actually, make it shrimp instead of salmon.",
     {("add", "contains shrimp", False), ("remove", "contains salmon", False)}),
    ("Suggest something healthy.", set()),
    ("Nothing that takes more than 20 minutes.", {("add", "time at most 20 min", False)}),
    ("Low in sodium, under 600 mg.", {("add", "sodium at most 600 mg", False)}),
    ("High protein dinner, at least 40 g.",
     {("add", "protein at least 40 g", False), ("add", "course: main-dish", False)}),
    ("I'm vegan and I can't eat gluten.", {("add", "vegan", True), ("add", "gluten-free", True)}),
    ("No eggs.", {("add", "egg-free", False)}),
    ("Swap broccoli for spinach.", {("add", "contains spinach", False), ("remove", "contains broccoli", False)}),
    ("Let's start over, I want to bake bread.", {("add", "course: breads", False)}),
])
def test_updates(ex, utterance, expected):
    assert updates(ex.extract(utterance)) == expected


def test_vegan_implies_no_separate_vegetarian(ex):
    assert updates(ex.extract("A vegan, vegetarian-friendly dinner")) == {
        ("add", "vegan", False), ("add", "course: main-dish", False)}


def test_objectives_only_without_numbers(ex):
    e = ex.extract("I want something quick but also high in protein.")
    assert e.objectives == [Objective("duration_min", "min"), Objective("protein_g", "max")]
    e = ex.extract("A quick lunch, ready in 20 minutes")
    assert e.objectives == []


def test_reset_detection(ex):
    assert ex.extract("Forget the dessert, I need a quick lunch instead").reset
    assert not ex.extract("Forget the protein requirement").reset


def test_removals_use_state(ex):
    state = DialogueState.from_active([
        {"kind": "min", "target": "protein_g", "value": 30, "source_turn": 1, "persistent": False},
        {"kind": "exclude", "target": "mushroom", "source_turn": 1, "persistent": False},
        {"kind": "diet", "target": "vegetarian", "source_turn": 1, "persistent": False}])
    assert updates(ex.extract("forget the protein requirement", state)) == {
        ("remove", "protein at least 30 g", False)}
    assert updates(ex.extract("Mushrooms are fine after all.", state)) == {
        ("remove", "no mushroom", False)}
    assert updates(ex.extract("It doesn't have to be vegetarian anymore.", state)) == {
        ("remove", "vegetarian", False)}
