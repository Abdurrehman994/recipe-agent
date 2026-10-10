"""Rule-based constraint extraction (baseline for the LLM extractor).

    extractor = RuleExtractor()
    extraction = extractor.extract("Vegetarian dinner under 25 minutes, no mushrooms", state)

The current DialogueState is passed in, as it is to the LLM extractor. Rules only use it to
resolve references such as "forget the protein requirement" (which constraint, which value).
The patterns follow the mapping rules in scenarios/README.md.
"""
from __future__ import annotations

import re
from collections import Counter

from src.data_prep import ALIASES, CATEGORIES
from src.schema import Constraint, ConstraintUpdate, Objective, TurnExtraction
from src.state import DialogueState
from src.tools import DB_PATH, stem

# ---------------------------------------------------------------- vocabularies
DIET_PATTERNS = {
    "vegan": r"\bvegan\b|\bplant[- ]based\b",
    "vegetarian": r"\bvegetarian\b|\bveggie\b|\bmeat[- ]?free\b|\bmeatless\b",
    "pescatarian": r"\bpesc[ae]tarian\b",
    "gluten_free": r"\bgluten[- ]free\b|\bno gluten\b|\bwithout gluten\b|\bc[o]?eliac\b|"
                   r"\bavoid(ing)? gluten\b|\bgluten (intolerance|allergy)\b|"
                   r"\b(can'?t|cannot|don'?t|do not|never) (eat|have|tolerate) gluten\b",
    "dairy_free": r"\bdairy[- ]free\b|\bno dairy\b|\bwithout dairy\b|\bnon[- ]dairy\b|"
                  r"\blactose\b|\bavoid(ing)? dairy\b|\bdairy (intolerance|allergy)\b|"
                  r"\b(can'?t|cannot|don'?t|do not|never) (eat|have|tolerate) dairy\b",
    "nut_free": r"\bnut[- ]free\b|\bno nuts\b|\bwithout nuts\b|\bnut allergy\b|"
                r"\ballergic to (pea)?nuts\b|\bavoid(ing)? nuts\b|\b(can'?t|cannot|don'?t|do not|never) (eat|have) nuts\b",
    "egg_free": r"\begg[- ]free\b|\bno eggs?\b|\bwithout eggs?\b|\begg allergy\b|"
                r"\ballergic to eggs?\b|\bavoid(ing)? eggs?\b|\b(can'?t|cannot|don'?t) (eat|have) eggs?\b",
}
# diet words that may appear in "X is fine now" style removals
DIET_WORDS = {"vegan": "vegan", "vegetarian": "vegetarian", "pescatarian": "pescatarian",
              "gluten": "gluten_free", "dairy": "dairy_free", "lactose": "dairy_free",
              "nuts": "nut_free", "nut": "nut_free", "eggs": "egg_free", "egg": "egg_free"}

COURSE_PATTERNS = [  # order matters: first match wins
    ("desserts", r"\bdesserts?\b|\bsomething sweet\b|\bsweet treat\b"),
    ("breakfast", r"\bbreakfast\b"),
    ("brunch", r"\bbrunch\b"),
    ("lunch", r"\blunch\b"),
    ("side-dishes", r"\bside( dish(es)?)?\b"),
    ("appetizers", r"\bappeti[sz]ers?\b|\bstarters?\b|\bfinger food\b"),
    ("soups-stews", r"\bsoups?\b|\bstews?\b"),
    ("salads", r"\bsalads?\b"),
    ("snacks", r"\bsnacks?\b"),
    ("breads", r"\bbake (some |a )?bread\b|\bbread recipe\b|\bloaf\b|\bhomemade bread\b"),
    ("main-dish", r"\bdinner\b|\bsupper\b|\bmain( (course|dish|meal))?\b|\bentr[ée]e\b"),
]

NUTRIENTS = {  # word pattern -> (field, unit pattern, default direction if no comparator)
    "calories": ("calories_kcal", r"(k?cal(orie)?s?|kcal)", "max"),
    "protein": ("protein_g", r"(g|grams?)", "min"),
    "sodium": ("sodium_mg", r"(mg|milligrams?)", "max"),
    "salt": ("sodium_mg", r"(mg|milligrams?)", "max"),
    "fat": ("fat_g", r"(g|grams?)", "max"),
    "saturated fat": ("saturated_fat_g", r"(g|grams?)", "max"),
    "sugar": ("sugar_g", r"(g|grams?)", "max"),
    "carbs": ("carbs_g", r"(g|grams?)", "max"),
    "carbohydrates": ("carbs_g", r"(g|grams?)", "max"),
    "fiber": ("fiber_g", r"(g|grams?)", "min"),
    "fibre": ("fiber_g", r"(g|grams?)", "min"),
    "cholesterol": ("cholesterol_mg", r"(mg|milligrams?)", "max"),
}
FIELD_WORDS = {"time": "duration_min", "minute": "duration_min", "duration": "duration_min",
               "cooking time": "duration_min", "time limit": "duration_min",
               **{w: f for w, (f, _, _) in NUTRIENTS.items()}, "calorie": "calories_kcal"}

MAX_WORDS = r"(under|less than|fewer than|below|at most|no more than|not more than|up to|" \
            r"max(imum)?|within|lower than|in|allow|only|takes?|ready in)"
MIN_WORDS = r"(at least|more than|over|above|minimum|min|no less than|not less than|higher than)"
NUM = r"(\d+(?:\.\d+)?)"
NEGATION = re.compile(r"\b(no|not|nothing|never|don'?t want|do not want)\b[^.,;]{0,25}$")


def _direction(cmp: str, before: str, default: str) -> str:
    """'more than 20 min' is a minimum, but 'nothing that takes more than 20 min' is a maximum."""
    if not cmp:
        return default
    kind = "min" if re.fullmatch(MIN_WORDS, cmp) else "max"
    if kind == "min" and NEGATION.search(before):
        kind = "max"
    return kind

OBJECTIVES = [
    (Objective("duration_min", "min"), r"\b(quick(est|er|ly)?|fast(est|er)?|speedy|in a hurry|"
                                       r"short on time|as soon as possible)\b"),
    (Objective("protein_g", "max"), r"\b(high(er)?[- ]protein|high(er)? in protein|protein[- ]rich|"
                                    r"(most|more|lots of|plenty of|enough|good amount of) protein|"
                                    r"care about protein|protein (is important|matters))\b"),
    (Objective("calories_kcal", "min"), r"\b(low(er)?[- ]cal(orie)?s?|low(er)? in calories|"
                                        r"fewest calories|(less|fewer) calories)\b"),
    (Objective("sodium_mg", "min"), r"\b(low(er)?[- ](sodium|salt)|less (salt|sodium))\b"),
    (Objective("fat_g", "min"), r"\b(low(er)?[- ]fat|less fat)\b"),
    (Objective("sugar_g", "min"), r"\b(low(er)?[- ]sugar|less sugar)\b"),
]

PERSISTENT_CUES = re.compile(
    r"\b(i'?m|i am|i've been|i have been|i follow|i keep|i eat|i don'?t eat|i do not eat|"
    r"i never eat|i can'?t eat|i cannot eat|i avoid|i'?m allergic|my diet|my family|"
    r"we'?re|we are|we don'?t eat|allergic|allergy|intolerant|intolerance)\b")
RESET = re.compile(
    r"\b(forget (it|that|this|everything|all that|about (it|that)|the (\w+ ){0,2}"
    r"(idea|dessert|dinner|lunch|breakfast|brunch|salad|soup|snack|meal|recipe|dish))|"
    r"never ?mind|start over|start again|something (completely |totally )?different|"
    r"new (request|question|search)|different question|change of plans?)\b")
EXCLUDE_TRIGGER = re.compile(
    r"\b(avoid(ing)?|without|no|skip|hold the|leave out|exclude|not|nothing with|hate|"
    r"dislike|don'?t (want|like|eat)|do not (want|like|eat)|never eat|can'?t (eat|have)|"
    r"cannot (eat|have)|allergic to|free of|minus|except)\s+(any\s+|the\s+)?")
INCLUDE_TRIGGER = re.compile(
    r"\b(with|contain(s|ing)?|using|use|made with|including|include|featuring|has|have|"
    r"based on|built around|some)\s+(some\s+|a\s+|an\s+|the\s+|fresh\s+)?")
DISH_WORDS = r"(dinner|lunch|breakfast|dish|recipe|meal|soup|salad|stew|curry|pasta|bake|" \
             r"casserole|stir[- ]?fry|tacos?|sandwich|burgers?|pie|dessert|cake|bowl)"
STAPLES = {"pasta", "rice", "noodle", "couscous", "quinoa", "tofu", "lentil", "chickpea",
           "potato", "egg", "spaghetti", "risotto"}
STOP_TERMS = {"time", "minute", "calorie", "dish", "recipe", "meal", "food", "something",
              "thing", "option", "idea", "dinner", "lunch", "breakfast", "dessert", "protein",
              "fat", "sugar", "sodium", "salt", "carb", "pepper", "water", "oil", "spice",
              "heat", "seasoning", "cooking", "sauce", "gluten", "dairy", "nut", "it",
              "one", "that", "this", "more", "less", "requirement", "limit", "side"}


def build_vocab(min_count: int = 5) -> set[str]:
    """Ingredient words/phrases users may name: frequent normalized ingredient names from the
    recipe database (stemmed), plus the keyword categories (mushroom, shellfish, ...)."""
    import json
    counts = Counter()
    with open(DB_PATH, encoding="utf-8") as f:
        for line in f:
            for name in json.loads(line)["ingredients"]:
                if len(name.split()) <= 3:
                    counts[" ".join(stem(w) for w in name.split())] += 1
    vocab = {n for n, c in counts.items() if c >= min_count and n.replace(" ", "").isalpha()}
    vocab |= {" ".join(stem(w) for w in k.split()) for k in ALIASES}  # user words: cilantro, ...
    vocab |= set(CATEGORIES) | {"meat", "fish", "seafood", "pork", "beef", "chicken",
                                "lamb", "tofu", "shrimp", "salmon", "tuna", "turkey"}
    return {v for v in vocab if v not in STOP_TERMS and len(v) > 2}


def _parse_duration(text: str) -> list[tuple[str, float, int, int]]:
    """Find time limits: returns (kind, minutes, start, end)."""
    found = []
    t = text.replace("half an hour", "30 minutes").replace("half hour", "30 minutes")
    t = re.sub(r"\ban hour and a half\b", "90 minutes", t)
    t = re.sub(r"\b(an|one) hour\b", "60 minutes", t)
    pattern = rf"(?:{MIN_WORDS}\s+|{MAX_WORDS}\s+)?{NUM}\s*(?:-\s*{NUM}\s*)?" \
              rf"(min(ute)?s?|hours?|hrs?|h)\b"
    for m in re.finditer(pattern, t):
        value = float(re.findall(NUM, m.group(0))[-1])  # for ranges "20-30 min" take the upper end
        unit = re.search(r"(min(ute)?s?|hours?|hrs?|h)$", m.group(0)).group(0)
        if unit.startswith("h"):
            value *= 60
        cmp = re.match(rf"{MIN_WORDS}|{MAX_WORDS}", m.group(0).strip())
        kind = _direction(cmp.group(0) if cmp else "", t[:m.start()], "max")
        found.append((kind, value, m.start(), m.end()))
    return found


def _parse_nutrients(text: str) -> list[tuple[str, str, float]]:
    """Find numeric nutrient limits in either order: 'under 500 calories', 'protein over 30 g'."""
    found, covered = [], []
    for word, (field, unit, default) in sorted(NUTRIENTS.items(), key=lambda x: -len(x[0])):
        w = word.replace(" ", r"\s+")
        unit_word = rf"{unit}?\s*(of\s+)?{w}" if field != "calories_kcal" else rf"({w}|kcal|cal)\b"
        patterns = [
            rf"(?P<cmp>{MIN_WORDS}|{MAX_WORDS})?\s*{NUM}\s*{unit_word}",                      # under 500 calories
            rf"\b{w}\s+(?:of\s+|should be\s+|is\s+|to\s+)?(?P<cmp>{MIN_WORDS}|{MAX_WORDS})?\s*{NUM}\s*{unit}?\b",  # protein over 30 g
        ]
        for p in patterns:
            for m in re.finditer(p, text):
                nums = re.findall(NUM, m.group(0))
                if not nums:
                    continue
                if field == "calories_kcal" and "cal" not in m.group(0) and "calorie" not in word:
                    continue
                cmp = (m.group("cmp") or "").strip()
                found.append((_direction(cmp, text[:m.start()], default), field, float(nums[-1])))
                covered.append(m.span())
        # "saturated fat" must not also count as "fat"
        text = re.sub(rf"\b{w}\b", " ", text) if word == "saturated fat" else text
    # bare amounts: "low in sodium, under 600 mg", "high protein dinner, at least 40 g"
    for m in re.finditer(rf"(?P<cmp>{MIN_WORDS}|{MAX_WORDS})?\s*(?P<num>\d+(?:\.\d+)?)\s*"
                         rf"(?P<unit>mg|milligrams?|g|grams?)\b", text):
        if any(a <= m.start("num") < b for a, b in covered):
            continue
        mg = m.group("unit").startswith("m")
        named = [NUTRIENTS[w] for w in NUTRIENTS if re.search(rf"\b{w}\b", text)
                 and NUTRIENTS[w][1].startswith("(mg" if mg else "(g")]
        if named:
            field, _, default = named[0]
        elif mg:
            field, default = "sodium_mg", "max"
        else:
            continue
        cmp = (m.group("cmp") or "").strip()
        found.append((_direction(cmp, text[:m.start()], default), field, float(m.group("num"))))
    unique = []
    for f in found:
        if f[1] not in [u[1] for u in unique]:
            unique.append(f)
    return unique


class RuleExtractor:
    name = "rules"

    def __init__(self, vocab: set[str] | None = None):
        self.vocab = vocab if vocab is not None else build_vocab()
        self.max_len = max(len(v.split()) for v in self.vocab)

    # ------------------------------------------------------------ helpers
    def find_terms(self, span: str) -> list[str]:
        """Longest-match ingredient terms from the vocabulary inside a short text span."""
        words = [stem(w) for w in re.findall(r"[a-z]+", span)]
        terms, i = [], 0
        while i < len(words):
            for n in range(min(self.max_len, len(words) - i), 0, -1):
                cand = " ".join(words[i:i + n])
                if cand in self.vocab:
                    terms.append(CANONICAL.get(cand, cand))
                    i += n
                    break
            else:
                i += 1
        return terms

    @staticmethod
    def _clause_after(text: str, start: int) -> str:
        """Text after a trigger up to the end of the clause."""
        rest = text[start:]
        m = re.search(r"[.;!?]|\b(but|so|because|that|which|please|if|in|for|under|over|"
                      r"with|ready|at|less|more|and (make|it|i|keep|something))\b", rest)
        return rest[: m.start()] if m else rest

    # ------------------------------------------------------------ main entry
    def extract(self, utterance: str, state: DialogueState | None = None) -> TurnExtraction:
        state = state or DialogueState()
        text = utterance.lower().replace("’", "'")
        updates: list[ConstraintUpdate] = []
        sentences = [s for s in re.split(r"(?<=[.!?;])\s+", text) if s.strip()]
        reset = bool(RESET.search(text))

        for sent in sentences:
            persistent = bool(PERSISTENT_CUES.search(sent))
            updates += self._removals(sent, state)
            removed_slots = {u.constraint.slot for u in updates if u.op == "remove"}

            # diets
            for diet, pat in DIET_PATTERNS.items():
                m = re.search(pat, sent)
                if m and not re.search(rf"(doesn'?t|does not|don'?t|no longer|not) (have to|need to)"
                                       rf" be\s+[\w-]*\s*{re.escape(m.group(0))}", sent):
                    if ("diet", diet) not in removed_slots:
                        updates.append(ConstraintUpdate("add", Constraint("diet", diet), persistent))
            if any(u.constraint.target == "vegan" and u.op == "add" for u in updates):
                updates = [u for u in updates if not (u.constraint.target == "vegetarian"
                                                      and u.op == "add")]

            # course
            for course, pat in COURSE_PATTERNS:
                if re.search(pat, sent) and not re.search(rf"(forget|instead of|not|no)\s+(the\s+|a\s+)?(?:{pat})", sent):
                    updates.append(ConstraintUpdate("add", Constraint("course", course), False))
                    break

            # numbers
            durations = _parse_duration(sent)
            for kind, value, *_ in durations:
                updates.append(ConstraintUpdate("add", Constraint(kind, "duration_min", value),
                                                persistent))
            for kind, field, value in _parse_nutrients(sent):
                updates.append(ConstraintUpdate("add", Constraint(kind, field, value), persistent))

            # ingredients
            updates += self._ingredients(sent, persistent, state)

        objectives = self._objectives(text, updates)
        return TurnExtraction(_dedupe(updates), objectives, reset)

    def _ingredients(self, sent: str, persistent: bool, state: DialogueState) -> list[ConstraintUpdate]:
        out = []
        diet_spans = [m.span() for pat in DIET_PATTERNS.values() for m in re.finditer(pat, sent)]
        excluded = set()
        for m in EXCLUDE_TRIGGER.finditer(sent):
            if any(a <= m.start() < b for a, b in diet_spans):
                continue
            span = self._clause_after(sent, m.end())
            for part in re.split(r",|\band\b|\bor\b|\bnor\b", span):
                for term in self.find_terms(part)[:1]:
                    if term in DIET_WORDS:
                        continue
                    excluded.add(term)
                    out.append(ConstraintUpdate("add", Constraint("exclude", term), persistent))
        # substitutions: "shrimp instead of salmon", "swap broccoli for spinach"
        swap = re.search(r"(?P<new>\w+(?: \w+)?) (instead of|rather than) (?:the )?(?P<old>\w+(?: \w+)?)", sent) \
            or re.search(r"\b(swap|replace|switch|change)\s+(?:the\s+)?(?P<old>\w+(?: \w+)?)\s+"
                         r"(for|with|to)\s+(?:some\s+)?(?P<new>\w+(?: \w+)?)", sent)
        if swap:
            new, old = self.find_terms(swap.group("new")), self.find_terms(swap.group("old"))
            if new:
                out.append(ConstraintUpdate("add", Constraint("include", new[-1]), persistent))
            if old:
                out.append(ConstraintUpdate("remove", Constraint("include", old[-1]), False))
            return out
        # "make it tofu instead": replaces the single included ingredient, if there is one
        swap = re.search(r"\b(make it|go with|use|switch to|try)\s+(?:some\s+)?(\w+(?: \w+)?)\s+instead\b", sent)
        if swap and self.find_terms(swap.group(2)):
            out.append(ConstraintUpdate("add", Constraint("include", self.find_terms(swap.group(2))[-1]),
                                        persistent))
            includes = state.find("include") if state else []
            if len(includes) == 1:
                out.append(ConstraintUpdate("remove", includes[0].constraint, False))
            return out
        candidates = []
        for m in INCLUDE_TRIGGER.finditer(sent):
            if re.match(r"(at least|no|out|less|more|under|over|\d)", sent[m.end():]):
                continue
            candidates += self.find_terms(self._clause_after(sent, m.end()))
        for m in re.finditer(rf"\b([a-z]+(?: [a-z]+)?)\s+{DISH_WORDS}\b", sent):
            candidates += self.find_terms(m.group(1))[-1:]
        # staples that name the dish itself: "pasta in under 30 minutes", "a rice bowl"
        # (but not inside a diet phrase such as "no eggs" or "allergic to eggs")
        diet_terms = {t for a, b in diet_spans for t in self.find_terms(sent[a:b])}
        candidates += [t for t in self.find_terms(sent) if t in STAPLES and t not in diet_terms]
        for term in candidates:
            if term not in excluded and term not in ("dairy", "gluten", "lactose"):
                out.append(ConstraintUpdate("add", Constraint("include", term), persistent))
        return out

    def _removals(self, sent: str, state: DialogueState) -> list[ConstraintUpdate]:
        out = []
        # "forget the protein requirement", "drop the time limit", "no calorie limit"
        for m in re.finditer(r"\b(forget|drop|remove|ignore|lift|skip|never ?mind|no need for|"
                             r"don'?t (care|worry) about|without the|no)\s+(about\s+)?(the\s+|my\s+|that\s+)?"
                             r"([a-z ]+?)\s+(requirement|limit|restriction|constraint|cap|target|"
                             r"minimum|maximum|rule)s?\b", sent):
            field = FIELD_WORDS.get(m.group(5).strip()) or FIELD_WORDS.get(m.group(5).strip().rstrip("s"))
            for a in state.active:
                c = a.constraint
                if field and c.target == field:
                    out.append(ConstraintUpdate("remove", c, False))
                elif not field and c.kind == "diet" and m.group(5).strip() in c.target:
                    out.append(ConstraintUpdate("remove", c, False))
        # "the calorie limit doesn't matter", "forget the protein", "drop the time"
        fields = []
        for m in re.finditer(r"\b(the\s+)?([a-z]+)\s+(limit|requirement|restriction|constraint)s?\s+"
                             r"(doesn'?t|does not|no longer) (matter|apply)|\b([a-z]+)\s+(doesn'?t|does not) matter",
                             sent):
            fields.append(m.group(2) or m.group(6))
        for m in re.finditer(r"\b(forget|drop|ignore|lift|skip|don'?t (care|worry) about)\s+(about\s+)?"
                             r"(the\s+|my\s+|that\s+)?([a-z]+)\b(?!\s+(requirement|limit|restriction|constraint))",
                             sent):
            fields.append(m.group(5))
        for word in fields:
            field = FIELD_WORDS.get(word) or FIELD_WORDS.get(word.rstrip("s"))
            for a in state.active:
                if field and a.constraint.target == field and a.constraint.kind in ("max", "min"):
                    out.append(ConstraintUpdate("remove", a.constraint, False))
        # "mushrooms are fine after all", "dairy is ok now", "I can have nuts again"
        for m in re.finditer(r"([a-z]+(?: [a-z]+)?)\s+(is|are)\s+(fine|ok(ay)?|allowed|alright|good)",
                             sent):
            word = m.group(1).split()[-1]
            if word in DIET_WORDS:
                for a in state.find("diet", DIET_WORDS[word]):
                    out.append(ConstraintUpdate("remove", a.constraint, False))
            for term in self.find_terms(m.group(1))[-1:]:
                for a in state.find("exclude", term):
                    out.append(ConstraintUpdate("remove", a.constraint, False))
        # "it doesn't have to be vegetarian anymore"
        for m in re.finditer(r"(doesn'?t|does not|don'?t|no longer|not) (have to|need to|necessarily)"
                             r"( be)?\s+([\w-]+)", sent):
            for diet, pat in DIET_PATTERNS.items():
                if re.search(pat, m.group(4)):
                    out.append(ConstraintUpdate("remove", Constraint("diet", diet), False))
        return out

    @staticmethod
    def _objectives(text: str, updates) -> list[Objective]:
        constrained = {u.constraint.target for u in updates if u.op == "add"}
        out = []
        for obj, pat in OBJECTIVES:
            if re.search(pat, text) and obj.target not in constrained:
                out.append(obj)
        return out


CANONICAL = {"mushroom": "mushroom", "nuts": "nuts", "seafood": "shellfish"}


def _dedupe(updates: list[ConstraintUpdate]) -> list[ConstraintUpdate]:
    seen, out = set(), []
    for u in updates:
        key = (u.op, u.constraint)
        if key not in seen:
            seen.add(key)
            out.append(u)
    return out
