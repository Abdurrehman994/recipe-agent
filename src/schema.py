"""Structured constraint schema shared by extractors, dialogue state, tools and evaluation.

A user requirement is a set of atomic constraints:

    Constraint("diet", "vegetarian")               vegetarian, vegan, pescatarian, gluten_free, ...
    Constraint("course", "main-dish")              main-dish, desserts, breakfast, ...
    Constraint("include", "chicken")               must contain an ingredient
    Constraint("exclude", "mushroom")              must not contain an ingredient / category
    Constraint("max", "duration_min", 25)          numeric upper bound (per serving for nutrients)
    Constraint("min", "protein_g", 30)             numeric lower bound

Atomic constraints make extraction F1 and dialogue-state accuracy plain set comparisons.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

DIETS = {"vegetarian", "pescatarian", "vegan", "gluten_free", "dairy_free", "nut_free", "egg_free"}
COURSES = {"main-dish", "side-dishes", "desserts", "breakfast", "lunch", "brunch", "appetizers",
           "salads", "soups-stews", "breads", "beverages", "snacks"}
NUMERIC_FIELDS = {  # field -> unit; all nutrients are per serving (see docs/data_notes.md)
    "duration_min": "min", "calories_kcal": "kcal", "protein_g": "g", "fat_g": "g",
    "saturated_fat_g": "g", "carbs_g": "g", "sugar_g": "g", "fiber_g": "g",
    "sodium_mg": "mg", "cholesterol_mg": "mg",
}
KINDS = {"diet", "course", "include", "exclude", "max", "min"}
# Kinds where a new value REPLACES the old one for the same target ("actually 30 min is fine").
# include/exclude/diet accumulate instead.
SINGLE_VALUE_KINDS = {"max", "min", "course"}


@dataclass(frozen=True)
class Constraint:
    kind: str
    target: str
    value: float | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown constraint kind: {self.kind}")
        if self.kind == "diet" and self.target not in DIETS:
            raise ValueError(f"unknown diet: {self.target}")
        if self.kind == "course" and self.target not in COURSES:
            raise ValueError(f"unknown course: {self.target}")
        if self.kind in ("max", "min"):
            if self.target not in NUMERIC_FIELDS:
                raise ValueError(f"unknown numeric field: {self.target}")
            if self.value is None:
                raise ValueError(f"{self.kind} constraint needs a value")
            object.__setattr__(self, "value", float(self.value))
        if self.kind in ("include", "exclude"):
            object.__setattr__(self, "target", self.target.lower().strip())

    @property
    def slot(self) -> tuple[str, str]:
        """Identity used for replacement: one max per field, one course, one entry per ingredient."""
        return (self.kind, self.target)

    def describe(self) -> str:
        if self.kind in ("max", "min"):
            unit = NUMERIC_FIELDS[self.target]
            name = self.target.rsplit("_", 1)[0] if self.target != "duration_min" else "time"
            word = "at most" if self.kind == "max" else "at least"
            return f"{name} {word} {self.value:g} {unit}"
        return {"diet": "{t}", "course": "course: {t}", "include": "contains {t}",
                "exclude": "no {t}"}[self.kind].format(t=self.target.replace("_", "-"))

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "target": self.target}
        if self.value is not None:
            d["value"] = self.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Constraint:
        return cls(d["kind"], d["target"], d.get("value"))


@dataclass(frozen=True)
class Objective:
    """A soft ranking preference, e.g. quickest or highest protein. Never filters recipes."""
    target: str          # a NUMERIC_FIELDS key
    direction: str       # "min" or "max"

    def __post_init__(self):
        if self.target not in NUMERIC_FIELDS or self.direction not in ("min", "max"):
            raise ValueError(f"bad objective: {self}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConstraintUpdate:
    """One change extracted from a user turn.

    op: "add" (new or replacing value for the same slot) or "remove" (drop a constraint).
    persistent: True for standing preferences ("I'm vegetarian", "I'm allergic to nuts"),
                False for requests tied to the current search ("tonight", "under 25 minutes").
    """
    op: str
    constraint: Constraint
    persistent: bool = False

    def to_dict(self) -> dict:
        return {"op": self.op, "constraint": self.constraint.to_dict(), "persistent": self.persistent}

    @classmethod
    def from_dict(cls, d: dict) -> ConstraintUpdate:
        return cls(d["op"], Constraint.from_dict(d["constraint"]), d.get("persistent", False))


@dataclass
class TurnExtraction:
    """Everything an extractor returns for one user utterance."""
    updates: list[ConstraintUpdate] = field(default_factory=list)
    objectives: list[Objective] = field(default_factory=list)
    reset: bool = False  # user starts a new, unrelated request ("forget that, ...")

    def to_dict(self) -> dict:
        return {"updates": [u.to_dict() for u in self.updates],
                "objectives": [o.to_dict() for o in self.objectives], "reset": self.reset}

    @classmethod
    def from_dict(cls, d: dict) -> TurnExtraction:
        return cls([ConstraintUpdate.from_dict(u) for u in d.get("updates", [])],
                   [Objective(**o) for o in d.get("objectives", [])], d.get("reset", False))
