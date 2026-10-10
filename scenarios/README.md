# Dialogue scenarios: format and annotation guidelines

- `dev/`: 30 scenarios for building and tuning (prompts, parser rules, policy).
- `test/`: 70 held-out scenarios. Do not look at system outputs on these until the final runs.

Each scenario has 3–5 scripted user turns. Turns are fixed: the next user turn is the same
whatever the agent says, so runs are reproducible. Check a split with:

```
python -m src.scenarios dev
```

The validator replays the gold updates and checks they produce the annotated `active` state
for every turn. It also counts the matching recipes in the database, and fails if a
`recommend` turn has 0 matches or a `report_infeasible` turn has any.

## File format

See [dev/dev_001.json](dev/dev_001.json) and the docstring in `src/scenarios.py`. Per turn:

| Field | Meaning |
|---|---|
| `user` | the utterance |
| `gold.updates` | what this turn changes: `add` / `remove` a constraint, with `persistent` |
| `gold.objectives` | soft ranking goals introduced in this turn (quickest, most protein, …) |
| `gold.reset` | the user abandons the current request ("forget that, …") |
| `active` | all constraints in force **after** this turn, with `source_turn` and `persistent` |
| `active_objectives` | objectives in force after this turn |
| `expected_action` | `recommend`, `clarify` or `report_infeasible` |

## Mapping rules (keep these consistent across all 100 scenarios)

**Meals → `course`**
- dinner, supper, main course, a meal → `main-dish`
- dessert, something sweet after dinner → `desserts`
- breakfast → `breakfast`; lunch → `lunch`; brunch → `brunch`
- side, side dish → `side-dishes`; starter, appetizer → `appetizers`
- soup, stew → `soups-stews`; salad → `salads`
- No meal word → no course constraint.

**Diets → `diet`** (use the diet flag, not `exclude`, for these)
- vegetarian, vegan, pescatarian, gluten-free, dairy-free / no dairy / lactose-free,
  nut-free / nut allergy, egg-free / no eggs.

**Ingredients → `include` / `exclude`**
- Use the singular base noun: "avoid mushrooms" → `exclude mushroom`; "with salmon" → `include salmon`.
- A category word that is not a diet (mushroom, shellfish, fish, meat) → `exclude <category>`.
- "Instead of X, Y" → `remove include X` + `add include Y`.

**Numbers → `max` / `min`** (nutrients are per serving)
- "under / less than / at most / no more than / up to N" → `max`. Treat "under 25 minutes"
  as ≤ 25, because the data uses round numbers.
- "at least / more than / over N" → `min`.
- Time: "half an hour" → 30, "an hour" → 60. "Quick" with no number is an **objective**
  (`duration_min` min), not a constraint.
- "High protein" with no number → objective (`protein_g` max); "low calorie" → objective
  (`calories_kcal` min). Only explicit numbers become constraints.

**Replacement vs addition**
- `max`, `min` and `course` hold one value per field: a new value replaces the old one
  (`add` with the new value; the validator drops the old one). `source_turn` becomes the
  new turn.
- `diet`, `include` and `exclude` accumulate. Lifting one needs an explicit `remove`
  ("mushrooms are fine after all").
- Adding `include X` removes an existing `exclude X`, and vice versa.

**Persistent vs temporary**
- `persistent: true` only for statements about the user: "I'm vegetarian", "I'm allergic
  to nuts", "I never eat pork", "I'm on a low-sodium diet, under 600 mg".
- Requests about the current dish are temporary: "a vegetarian dinner", "it has to be
  gluten-free", "under 25 minutes".
- `reset: true` when the user abandons the current request ("forget the dessert",
  "something different: …", "new question"). Reset drops all temporary constraints and
  objectives but keeps persistent ones.

**Expected action**
- `clarify`: the request cannot be turned into constraints ("something healthy",
  "something light", "a nice meal") **and** asking is reasonable. The next scripted turn
  answers the question.
- `report_infeasible`: no recipe in the database satisfies the active constraints. The
  validator enforces this. The agent should say so and explain which constraint blocks it,
  without dropping one silently.
- `recommend`: everything else.

## Target coverage (100 scenarios)

| Phenomenon | Approx. scenarios | Example |
|---|---|---|
| add constraints over turns | most | vegetarian → under 25 min → no mushrooms |
| replace a numeric limit | 25+ | "actually 30 minutes is fine" |
| remove a constraint | 15+ | "mushrooms are fine after all" |
| persistent preference + reset | 15+ | nut allergy survives "forget the dessert" |
| infeasible combination | 20+ | vegan + gluten-free + 10 min + 30 g protein |
| explicit relaxation after infeasible | 10+ | "ok, allow 30 minutes" |
| clarification needed | 10+ | "something healthy" |
| competing objectives / trade-off | 10+ | quickest vs highest protein |

Use the same proportions in dev (30) and test (70).
