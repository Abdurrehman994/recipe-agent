"""Dialogue state (structured memory).

Keeps every active constraint together with where it came from:
    source_turn  the user turn that introduced (or last changed) it
    persistent   a standing preference about the user ("I'm vegan") vs. a request about the
                 current dish ("a vegan dinner")

Update semantics (also the reference used to validate scenario annotations):
    reset   drops temporary constraints and objectives, keeps persistent constraints
    add     max/min/course replace the old value for the same field;
            diet/include/exclude accumulate; include X cancels exclude X and vice versa
    remove  drops the constraint for that slot (value is ignored)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.schema import SINGLE_VALUE_KINDS, Constraint, Objective, TurnExtraction


@dataclass
class ActiveConstraint:
    constraint: Constraint
    source_turn: int
    persistent: bool = False

    def to_dict(self) -> dict:
        return {**self.constraint.to_dict(), "source_turn": self.source_turn,
                "persistent": self.persistent}

    @classmethod
    def from_dict(cls, d: dict) -> ActiveConstraint:
        return cls(Constraint.from_dict(d), d["source_turn"], d.get("persistent", False))


@dataclass
class DialogueState:
    active: list[ActiveConstraint] = field(default_factory=list)
    objectives: list[Objective] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)  # what changed in each turn, for logs

    @property
    def constraints(self) -> list[Constraint]:
        return [a.constraint for a in self.active]

    def find(self, kind: str | None = None, target: str | None = None) -> list[ActiveConstraint]:
        return [a for a in self.active
                if (kind is None or a.constraint.kind == kind)
                and (target is None or a.constraint.target == target)]

    def apply(self, ext: TurnExtraction, turn: int) -> DialogueState:
        changes = []
        if ext.reset:
            dropped = [a for a in self.active if not a.persistent]
            self.active = [a for a in self.active if a.persistent]
            self.objectives = []
            changes += [("reset", a.constraint.describe()) for a in dropped]
        for u in ext.updates:
            c = u.constraint
            flip = {"include": "exclude", "exclude": "include"}.get(c.kind)

            def replaced(a: ActiveConstraint) -> bool:
                old = a.constraint
                if old.slot == c.slot:
                    return c.kind in SINGLE_VALUE_KINDS or u.op == "remove" or old.value == c.value
                return u.op == "add" and flip is not None and old.slot == (flip, c.target)

            gone = [a for a in self.active if replaced(a)]
            self.active = [a for a in self.active if not replaced(a)]
            changes += [("removed" if u.op == "remove" else "replaced", a.constraint.describe())
                        for a in gone]
            if u.op == "add":
                self.active.append(ActiveConstraint(c, turn, u.persistent))
                changes.append(("added", c.describe()))
        for o in ext.objectives:
            self.objectives = [x for x in self.objectives if x.target != o.target] + [o]
            changes.append(("objective", f"{o.direction} {o.target}"))
        self.history.append({"turn": turn, "changes": changes})
        return self

    def to_active(self) -> list[dict]:
        return [a.to_dict() for a in self.active]

    @classmethod
    def from_active(cls, active: list[dict], objectives=()) -> DialogueState:
        return cls([ActiveConstraint.from_dict(a) for a in active],
                   [o if isinstance(o, Objective) else Objective(**o) for o in objectives])

    def describe(self) -> str:
        if not self.active and not self.objectives:
            return "no constraints yet"
        parts = [f"{a.constraint.describe()} (turn {a.source_turn}"
                 f"{', standing preference' if a.persistent else ''})" for a in self.active]
        parts += [f"objective: {'lowest' if o.direction == 'min' else 'highest'} {o.target}"
                  for o in self.objectives]
        return "; ".join(parts)
