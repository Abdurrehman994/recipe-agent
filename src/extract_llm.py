"""LLM-based constraint extraction (compared against the rule baseline).

    extractor = LLMExtractor()                      # Claude, needs ANTHROPIC_API_KEY
    extractor = LLMExtractor(provider="groq")       # free tier, needs GROQ_API_KEY
    extraction = extractor.extract("Vegetarian dinner under 25 minutes, no mushrooms", state)

The model sees the system prompt (schema + the mapping rules from scenarios/README.md), the
current dialogue state and the new user turn, and returns JSON constrained by a schema
(structured outputs). The JSON is validated against src/schema.py; invalid items are dropped
and logged in `self.dropped` rather than crashing the evaluation.

Providers
  anthropic  Claude via the `anthropic` SDK (default model claude-opus-5-5).
  groq       open-weight models on Groq's free tier via the `groq` SDK (default model
             openai/gpt-oss-120b, which supports JSON-schema output). The free tier allows
             30 requests/min and ~1,000/day per organisation (see
             console.groq.com/settings/limits), so calls are paced to stay under the
             per-minute limit and 429s are retried by the SDK.

Responses are cached on disk (.cache/llm/) keyed by provider, model, effort, prompt version,
state and utterance, so re-running the evaluation is free and reproducible. Bump
PROMPT_VERSION whenever SYSTEM_PROMPT changes.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from src.schema import (COURSES, DIETS, KINDS, NUMERIC_FIELDS, Constraint, ConstraintUpdate,
                        Objective, TurnExtraction)
from src.state import DialogueState
from src.tools import stem

DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "groq": "openai/gpt-oss-120b"}
MODEL = DEFAULT_MODELS["anthropic"]
GROQ_MIN_INTERVAL_S = 2.1  # 30 requests/min on the free tier
PROMPT_VERSION = 2
CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "llm"
load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # GROQ_API_KEY / ANTHROPIC_API_KEY
CANONICAL = {"seafood": "shellfish"}

SYSTEM_PROMPT = f"""You convert one user turn of a recipe-recommendation dialogue into structured \
constraint updates. You are given the constraints already active and the new user message. \
Return only what THIS turn changes; do not repeat constraints that are already active and \
unchanged.

## Output
- updates: list of {{op, kind, target, value, persistent}}
  - op: "add" (new constraint, or a new value replacing the old one for the same field) or \
"remove" (lift an active constraint; copy kind and target from the active list, value null)
  - kind / target:
    - diet: one of {sorted(DIETS)}
    - course: one of {sorted(COURSES)}
    - include / exclude: an ingredient as a lowercase singular base noun ("mushroom", "tomato")
    - max / min: one of {sorted(NUMERIC_FIELDS)} (nutrients per serving), with a numeric value
  - value: number for max/min, null otherwise
  - persistent: true only for standing statements about the user ("I'm vegetarian", "I'm \
allergic to nuts", "I never eat pork", "I'm on a low-sodium diet, under 600 mg"). Requests \
about the current dish are false ("a vegetarian dinner", "it has to be gluten-free").
- objectives: soft ranking goals introduced in this turn, {{target, direction}} with direction \
"min" or "max": "quick" -> duration_min min; "high protein" -> protein_g max; "low calorie" -> \
calories_kcal min; "low sodium/salt" -> sodium_mg min; "low fat" -> fat_g min; "low sugar" -> \
sugar_g min. Only when no explicit number is given; an explicit number is a max/min constraint. \
Comparative questions about the candidates introduce one objective per compared attribute: \
"which is faster and which has more protein?" -> duration_min min + protein_g max; "what's \
the quickest?" -> duration_min min.
- reset: true when the user abandons the current request ("forget that, ...", "forget the \
dessert, ...", "never mind the side, ...", "something different: ...", "let's start over", "new \
question"). Reset itself drops every temporary constraint and objective and keeps persistent \
ones, so with reset: true emit NO remove updates; list only the adds of the new request. \
"Forget the dessert, I need a quick lunch instead" -> reset true, add course lunch, objective \
duration_min min. Use remove only when the user lifts one constraint while the request \
continues ("mushrooms are fine after all").

## Mapping rules
Meals -> course: dinner, supper, main course, a meal -> main-dish; dessert, something sweet \
after dinner -> desserts; breakfast -> breakfast; lunch -> lunch; brunch -> brunch; side, side \
dish -> side-dishes; starter, appetizer -> appetizers; soup, stew -> soups-stews; salad -> \
salads. No meal word -> no course; "tonight", "today", "this evening", "for the kids" are not \
meal words. Only one course is active at a time: a new meal word is an add that replaces the \
old course, so never emit a remove for the previous course ("oh wait, I meant dinner" -> add \
course main-dish only). The same holds for max/min: a new number is an add, with no remove.

Diets: vegetarian, vegan, pescatarian, gluten-free, dairy-free / no dairy / lactose-free, \
nut-free / nut allergy, egg-free / no eggs are diet constraints, never exclude. Allergies and \
intolerances to these categories are diets: "I'm allergic to eggs" -> diet egg_free \
persistent; "nut allergy" -> diet nut_free persistent; "lactose intolerant" -> diet \
dairy_free persistent. Only an allergy to a single ingredient outside these categories is an \
exclude ("allergic to shellfish" -> exclude shellfish persistent). Vegan implies vegetarian: \
output only vegan.

Ingredients: "avoid mushrooms" -> exclude mushroom; "with salmon" -> include salmon. A category \
that is not a diet (mushroom, shellfish, fish, meat) -> exclude <category>; seafood -> \
shellfish. "Y instead of X" -> remove include X + add include Y. Staples that name the dish \
(pasta, rice, noodles, tofu, lentils, potatoes, eggs) -> include ("pasta in under 30 minutes" \
-> include pasta). Other dish types (curry, stir-fry, tacos) are not constraints.

Numbers: "under / less than / at most / no more than / up to / within N" -> max; "at least / \
more than / over N" -> min, except after a negation ("nothing that takes more than 20 \
minutes" -> max). Time in minutes: half an hour -> 30, an hour -> 60. "Under 25 minutes" -> \
max 25. A new number for a field already limited is an add with the new value.

Removals: "mushrooms are fine after all" -> remove exclude mushroom; "forget the calorie \
limit" -> remove the active calories_kcal constraint; "it doesn't have to be vegetarian" -> \
remove diet vegetarian. Only remove constraints that are in the active list.

Vague requests that cannot be turned into constraints ("something healthy", "something light \
for tonight", "a nice meal") produce no updates and no objectives. When a later turn then names \
a course or number that is not yet active, it is an add even if it seems implied earlier."""

SCHEMA = {
    "type": "object",
    "properties": {
        "updates": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["add", "remove"]},
                "kind": {"type": "string", "enum": sorted(KINDS)},
                "target": {"type": "string"},
                "value": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                "persistent": {"type": "boolean"},
            },
            "required": ["op", "kind", "target", "value", "persistent"],
            "additionalProperties": False,
        }},
        "objectives": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "enum": sorted(NUMERIC_FIELDS)},
                "direction": {"type": "string", "enum": ["min", "max"]},
            },
            "required": ["target", "direction"],
            "additionalProperties": False,
        }},
        "reset": {"type": "boolean"},
    },
    "required": ["updates", "objectives", "reset"],
    "additionalProperties": False,
}


def _normalize_ingredient(target: str) -> str:
    words = " ".join(stem(w) for w in target.lower().strip().split())
    return CANONICAL.get(words, words)


def parse_output(data: dict, state: DialogueState | None = None) -> tuple[TurnExtraction, list[str]]:
    """Validate model JSON into a TurnExtraction. Returns (extraction, dropped item messages)."""
    state = state or DialogueState()
    updates, dropped = [], []
    for u in data.get("updates", []):
        target = u["target"]
        if u["kind"] in ("include", "exclude"):
            target = _normalize_ingredient(target)
        value = u.get("value") if u["kind"] in ("max", "min") else None
        if u["op"] == "remove" and u["kind"] in ("max", "min") and value is None:
            # removals ignore the value, but Constraint needs one: take it from the state
            active = state.find(u["kind"], target)
            value = active[0].constraint.value if active else None
        try:
            c = Constraint(u["kind"], target, value)
        except ValueError as e:
            dropped.append(f"{u}: {e}")
            continue
        updates.append(ConstraintUpdate(u["op"], c, bool(u.get("persistent")) and u["op"] == "add"))
    objectives = []
    for o in data.get("objectives", []):
        try:
            objectives.append(Objective(o["target"], o["direction"]))
        except ValueError as e:
            dropped.append(f"{o}: {e}")
    return TurnExtraction(updates, objectives, bool(data.get("reset"))), dropped


def render_state(state: DialogueState) -> str:
    active = [{k: v for k, v in a.to_dict().items() if k != "source_turn"} for a in state.active]
    objectives = [o.to_dict() for o in state.objectives]
    return json.dumps({"active_constraints": active, "active_objectives": objectives})


class LLMExtractor:
    def __init__(self, model: str | None = None, effort: str = "medium", client=None,
                 cache_dir: Path | None = CACHE_DIR, provider: str = "anthropic",
                 min_interval_s: float | None = None):
        if provider not in DEFAULT_MODELS:
            raise ValueError(f"unknown provider {provider!r}; choose from {sorted(DEFAULT_MODELS)}")
        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.effort = effort
        self.cache_dir = cache_dir
        self._client = client
        if min_interval_s is None:
            min_interval_s = GROQ_MIN_INTERVAL_S if provider == "groq" else 0.0
        self.min_interval_s = min_interval_s
        self._last_call = 0.0
        self.dropped: list[str] = []  # schema violations, for error analysis

    @property
    def name(self) -> str:
        """Used in result file names: llm-<provider>-<model>."""
        return f"llm-{self.provider}-{self.model.split('/')[-1]}"

    @property
    def client(self):
        if self._client is None:
            if self.provider == "groq":
                import groq
                self._client = groq.Groq(max_retries=5)  # retries 429s using retry-after
            else:
                import anthropic
                self._client = anthropic.Anthropic()
        return self._client

    def extract(self, utterance: str, state: DialogueState | None = None) -> TurnExtraction:
        state = state or DialogueState()
        user = f"Current state:\n{render_state(state)}\n\nUser message:\n{utterance}"
        data = self._cached_call(user)
        ext, dropped = parse_output(data, state)
        self.dropped += [f"{utterance!r}: {d}" for d in dropped]
        return ext

    def _cached_call(self, user: str) -> dict:
        key = hashlib.sha256(json.dumps(
            [self.provider, self.model, self.effort, PROMPT_VERSION, user]).encode()).hexdigest()[:24]
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        data = self._call(user)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def _call(self, user: str) -> dict:
        if self.min_interval_s:  # pace requests for rate-limited free tiers
            wait = self._last_call + self.min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
        if self.provider == "groq":
            return self._call_groq(user)
        return self._call_anthropic(user)

    def _call_groq(self, user: str) -> dict:
        kwargs = {}
        if "gpt-oss" in self.model:  # only the gpt-oss models accept a reasoning effort
            kwargs["reasoning_effort"] = self.effort
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "turn_extraction", "schema": SCHEMA}},
            temperature=0,
            max_completion_tokens=4000,
            **kwargs,
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("groq output truncated (max_completion_tokens)")
        return json.loads(choice.message.content)

    def _call_anthropic(self, user: str) -> dict:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            cache_control={"type": "ephemeral"},
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {response.stop_details}")
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)
