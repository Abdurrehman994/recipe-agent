import copy

import pytest

from src.scenarios import apply_extraction, load_scenarios, validate
from src.schema import Constraint, ConstraintUpdate, TurnExtraction
from src.tools import RecipeDB


def ext(*updates, reset=False):
    return TurnExtraction([ConstraintUpdate(op, Constraint(*c), p) for op, c, p in updates],
                          reset=reset)


def test_replace_single_value_slot():
    s = apply_extraction([], ext(("add", ("max", "duration_min", 25), False)), 1)
    s = apply_extraction(s, ext(("add", ("max", "duration_min", 30), False)), 2)
    assert s == [{"kind": "max", "target": "duration_min", "value": 30.0,
                  "source_turn": 2, "persistent": False}]


def test_include_and_exclude_accumulate_and_flip():
    s = apply_extraction([], ext(("add", ("exclude", "mushroom"), False),
                                 ("add", ("exclude", "onion"), False)), 1)
    assert len(s) == 2
    s = apply_extraction(s, ext(("add", ("include", "onion"), False)), 2)
    assert {(a["kind"], a["target"]) for a in s} == {("exclude", "mushroom"), ("include", "onion")}


def test_reset_keeps_only_persistent():
    s = apply_extraction([], ext(("add", ("diet", "nut_free"), True),
                                 ("add", ("course", "desserts"), False)), 1)
    s = apply_extraction(s, ext(("add", ("course", "lunch"), False), reset=True), 2)
    assert {(a["target"], a["source_turn"]) for a in s} == {("nut_free", 1), ("lunch", 2)}


@pytest.fixture(scope="module")
def db():
    return RecipeDB.load()


def test_dev_scenarios_are_valid(db):
    scenarios = load_scenarios("dev")
    assert scenarios
    for sc in scenarios:
        assert validate(sc, db) == [], sc["id"]


def test_validator_catches_annotation_error(db):
    sc = copy.deepcopy(load_scenarios("dev")[0])
    sc["turns"][3]["active"][2]["value"] = 25  # forgot to apply the replacement
    errors = validate(sc, db)
    assert any("does not match replayed updates" in e for e in errors)
