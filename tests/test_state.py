from src.schema import Constraint, ConstraintUpdate, Objective, TurnExtraction
from src.state import DialogueState


def ext(*updates, objectives=(), reset=False):
    return TurnExtraction([ConstraintUpdate(op, Constraint(*c), p) for op, c, p in updates],
                          list(objectives), reset)


def test_replacement_keeps_provenance_of_new_turn():
    s = DialogueState()
    s.apply(ext(("add", ("max", "duration_min", 25), False)), 1)
    s.apply(ext(("add", ("max", "duration_min", 30), False)), 4)
    assert s.to_active() == [{"kind": "max", "target": "duration_min", "value": 30.0,
                              "source_turn": 4, "persistent": False}]
    assert ("replaced", "time at most 25 min") in s.history[-1]["changes"]


def test_persistent_survives_reset_objectives_do_not():
    s = DialogueState()
    s.apply(ext(("add", ("diet", "vegan"), True), ("add", ("course", "desserts"), False),
                objectives=[Objective("duration_min", "min")]), 1)
    s.apply(ext(("add", ("course", "lunch"), False), reset=True), 2)
    assert [(a.constraint.target, a.persistent) for a in s.active] == [("vegan", True), ("lunch", False)]
    assert s.objectives == []


def test_objective_direction_is_replaced():
    s = DialogueState()
    s.apply(ext(objectives=[Objective("calories_kcal", "min")]), 1)
    s.apply(ext(objectives=[Objective("calories_kcal", "max")]), 2)
    assert s.objectives == [Objective("calories_kcal", "max")]


def test_remove_ignores_value():
    s = DialogueState()
    s.apply(ext(("add", ("min", "protein_g", 30), False)), 1)
    s.apply(ext(("remove", ("min", "protein_g", 0), False)), 2)
    assert s.active == []


def test_describe():
    s = DialogueState()
    assert s.describe() == "no constraints yet"
    s.apply(ext(("add", ("diet", "nut_free"), True)), 1)
    assert s.describe() == "nut-free (turn 1, standing preference)"
