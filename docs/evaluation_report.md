# Evaluation report: Conversational recipe agent with memory and tool use

Project 25, NLP course, University of Oulu, autumn 2025. Final evaluation run 2026-10-23.

**Research question.** Do structured memory and explicit tool use improve constraint
satisfaction as user requirements change during a conversation?

**Answer.** Yes, and the two contributions are separable. On 70 held-out dialogues, a
conversational RAG baseline satisfies all active user constraints for 25 % of its
recommendations. Adding structured memory raises this to 83–91 % (depending on the extractor).
Adding deterministic validation and a decision policy on top raises it to 89–99 %, and reduces
silent relaxations of explicit constraints from 237 turns to 3. Removing memory from the full
agent drops satisfaction to 43 %; removing validation drops it to 26 %. Every remaining failure
of the full agent on held-out data traces to a constraint-extraction error upstream, not to
the policy: on turns where the extracted state was correct, the agent never relaxed a
constraint.

## 1. Setup

### Data

5,000 recipes sampled from the HUMMUS dataset (507,335 rows) after quality filters
(`docs/data_notes.md`). Nutrition values were verified to be per serving, not per 100 g, by
five independent checks (for example, 22.5 % of recipes have fat + carbs + protein above
100 g, which is impossible per 100 g). The raw `ingredient_food_kg_names` column was found to be
misaligned with its rows and is not used; ingredient matching runs on the free-text ingredient
lines. Durations are in minutes.

### Dialogue scenarios

100 scripted scenarios of 3–5 user turns (375 turns): 30 development (`scenarios/dev`, used to
build and tune every component) and 70 held-out (`scenarios/test`, used once, for the numbers
in this report). Every turn is annotated with the constraint updates it introduces, the full
set of active constraints afterwards (with origin turn and persistent flag), active ranking
objectives, and the expected agent action (`recommend`, `clarify`, `report_infeasible`).
A validator replays the annotations through the memory rules and checks feasibility against
the database, so every `report_infeasible` turn has zero matching recipes and every
`recommend` turn has at least one. Held-out coverage: 19 infeasible combinations, 13
relaxations after an infeasible request, 12 clarification turns, 11 competing-objective
turns, 16 persistent preferences, 11 resets, 10 removals, 23 replacements of a value or course.

User turns are fixed in advance, so every run is reproducible and the comparison between
systems is paired.

### Constraint schema and extraction

A user requirement is a set of atomic constraints: `diet` (vegetarian, vegan, pescatarian,
gluten-free, dairy-free, nut-free, egg-free), `course`, `include`/`exclude` ingredient,
`max`/`min` on duration or a nutrient per serving. Turns can also carry soft objectives
(for example lowest duration, highest protein) and a reset. Two extractors:

- **Rules**: a hand-written parser (keyword lexicons, numeric patterns, negation handling),
  tuned on the development split.
- **LLM**: `openai/gpt-oss-120b` (open weights), reasoning effort medium, temperature 0,
  JSON-schema-constrained output. The prompt contains the schema and the annotation
  guidelines' mapping rules, and was tuned on the development split in two iterations. Served
  by Lehmus AI (University of Oulu) for the final run; Groq's free tier was used for the pilot
  and gives the same development results. Every response is validated against the schema.

### Dialogue state (memory)

Active constraints with origin turn and a persistent flag. `max`/`min`/`course` hold one value
per field: a new value replaces the old one ("actually, 30 minutes is fine"). `diet`,
`include`, `exclude` accumulate and need an explicit removal. `include X` cancels `exclude X`
and vice versa. A reset drops temporary constraints and objectives but keeps persistent ones
("I'm allergic to nuts" survives "forget the dessert").

### Tools (deterministic code, no model judgement)

`search_recipes` (hard filter by constraints, then rank by objectives, BM25 relevance and
rating), `get_recipe_details`, `validate_constraints` (per-constraint pass/fail with evidence
from the data), `diagnose_constraints` (for an over-constrained request: matches if each
constraint were dropped, and the closest available numeric value).

### Systems compared

| System | State | Retrieval | Validation / policy |
|---|---|---|---|
| RAG baseline | none (raw transcript) | BM25 over the transcript | none; always recommends |
| RAG + structured memory | extracted constraints | hard-filtered by the state | none; if nothing matches, falls back to unfiltered retrieval (silent relaxation) |
| Full agent | extracted constraints | hard-filtered by the state | clarifies when nothing can be constrained; inspects and validates each candidate; reports infeasibility with a diagnosis; never relaxes a constraint |
| Agent, no memory | rebuilt from the current turn only | as agent | as agent |
| Agent, no validation | extracted constraints | constraints only as soft query words | no validation; cannot detect infeasibility |

Explanations are generated from tool output (recipe facts, per-constraint evidence, the
diagnosis for infeasible requests, and the quickest-versus-highest-X trade-off when two
objectives compete). Every factual statement is also emitted as a structured claim and
re-checked against the database.

### Metrics

Extraction: micro precision/recall/F1 over atomic update items per turn. State: exact-match
accuracy per turn against the annotated active set (constraints and objectives), plus
provenance accuracy (origin turn and persistent flag). Systems: action accuracy, recall of
clarify and infeasible turns, constraint satisfaction (share of recommended recipes satisfying
all *gold* active constraints), silent relaxation rate (recommend turns where a recommendation
violates a gold constraint or no recipe could satisfy them), unsupported factual claims, and
tool calls per turn.

## 2. Results

### 2.1 Constraint extraction and dialogue state

| Extractor | Split | Extraction F1 | Precision | Recall | State exact | State F1 | Provenance acc. |
|---|---|---|---|---|---|---|---|
| Rules | dev | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.990 |
| LLM | dev | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 |
| Rules | **held-out** | **0.936** | 0.967 | 0.907 | **0.783** | 0.952 | 0.968 |
| LLM | **held-out** | **0.968** | 0.969 | 0.966 | **0.925** | 0.974 | 0.987 |

Both extractors are perfect on the split they were tuned on and both degrade on held-out
dialogues, the rules twice as much. The gap is largest in recall: the rule parser misses
phrasings its lexicon does not cover (five missed resets, five missed inclusions, four missed
courses, six missed objectives on held-out turns), while the LLM's remaining errors are
spread thinly (four missed exclusions, three missed resets, four spurious diets). Because
every extraction error propagates into the state and persists across later turns, a 3-point
F1 gap becomes a 14-point gap in state accuracy.

Memory ablation at the state level (rules, held-out): rebuilding the state from the current
turn alone gives 0.273 exact match, against 0.783 with memory.

### 2.2 Systems, held-out split (70 scenarios, 267 turns)

With the rule-based extractor:

| System | Action acc. | Clarify recall | Infeasible recall | Constraint satisfaction | Turns fully satisfied | Silent relaxations | Tool calls / turn |
|---|---|---|---|---|---|---|---|
| RAG baseline | 0.884 | 0.00 | 0.00 | 0.248 | 0.112 | 237 (89 %) | 1.0 |
| RAG + memory | 0.884 | 0.00 | 0.00 | 0.834 | 0.813 | 50 (19 %) | 1.1 |
| **Full agent** | **0.951** | 0.75 | **0.89** | **0.893** | **0.875** | **29 (12 %)** | 6.2 |
| Agent, no memory | 0.820 | 1.00 | 0.21 | 0.406 | 0.353 | 141 (65 %) | 5.7 |
| Agent, no validation | 0.906 | 0.75 | 0.00 | 0.259 | 0.122 | 224 (88 %) | 1.0 |

With the LLM extractor (same scenarios, same policy):

| System | Action acc. | Clarify recall | Infeasible recall | Constraint satisfaction | Turns fully satisfied | Silent relaxations | Tool calls / turn |
|---|---|---|---|---|---|---|---|
| RAG baseline | 0.884 | 0.00 | 0.00 | 0.248 | 0.112 | 237 (89 %) | 1.0 |
| RAG + memory | 0.884 | 0.00 | 0.00 | 0.911 | 0.910 | 24 (9 %) | 1.1 |
| **Full agent** | **0.974** | 0.75 | **0.95** | **0.990** | **0.987** | **3 (1 %)** | 6.3 |
| Agent, no memory | 0.835 | 1.00 | 0.21 | 0.434 | 0.383 | 137 (62 %) | 5.8 |
| Agent, no validation | 0.918 | 0.75 | 0.00 | 0.257 | 0.120 | 227 (88 %) | 1.0 |

Unsupported factual claims: 0 of 25,000+ claims checked across all runs (explanations are
generated from tool output, and the verifier re-checks each claim against the database).

The RAG baseline's action accuracy of 0.884 is the share of turns whose expected action is
`recommend`; it never clarifies or reports infeasibility. The memory ablation's clarify recall
of 1.00 is an artefact: with no memory, any vague turn has an empty state and triggers a
clarification.

### 2.3 Where the full agent's remaining failures come from

Each silent relaxation and wrong action of the full agent on held-out data was classified by
whether the extracted state matched the gold state on that turn:

| Extractor | Silent relaxations, state wrong | Silent relaxations, state correct | Wrong actions, state wrong | Wrong actions, state correct |
|---|---|---|---|---|
| Rules | 29 | **0** | 11 | 2 |
| LLM | 3 | **0** | 5 | 2 |

With a correct state the policy never relaxed a constraint. The two residual action errors
(both extractors, same two turns) are the one policy weakness found: when a persistent
preference is already active ("I'm vegetarian") and the user then asks something vague ("What
should I cook?"), the agent searches instead of asking, because it clarifies only when no
constraint is active. This case does not occur in the development split. It is left as
future work rather than fixed, since fixing it now would mean tuning on the held-out data.

By contrast, RAG + memory with the LLM extractor relaxes on 17 turns where its state was
correct: all 17 are infeasible requests where it fell back to unfiltered retrieval and
recommended anyway. That is exactly the behaviour the validation step and the policy prevent.

### 2.4 Infeasible and clarification handling

On the 19 held-out infeasible turns the full agent reports infeasibility in 17 (rules) or 18
(LLM) cases; each report names the most restrictive requirement, how many recipes would match
without it, and the closest available numeric value, without dropping anything. The misses are
turns where an extraction error produced a state that *was* feasible. Baselines never report
infeasibility. On the 12 clarification turns, the agent asks in 9; the other 3 are the
persistent-preference case above.

### 2.5 Cost and tool usage

The LLM is used for extraction only, one call per user turn; retrieval, validation, policy and
explanations are deterministic. Measured on Lehmus AI: 1,739 tokens per call on held-out turns
(1,400 prompt, 340 completion including the model's reasoning), about 7 s latency. The full
held-out evaluation used 534 model calls (0.93 M tokens); the development verification pass a
further 108 calls. The full agent makes 6.3 tool calls per turn (one search, then details and
validation for each inspected candidate, plus a diagnosis on infeasible turns); the baselines
make one.

## 3. Discussion

**Memory and validation contribute separately and additively.** Memory is what makes a
multi-turn request coherent: without it, "under 25 minutes" on turn 2 forgets "vegetarian" from
turn 1, and satisfaction halves. Validation is what makes the recommendation honest: with
retrieval only (no validation), the system still answers every turn, but 88 % of its answers
violate something the user said, and infeasible requests are never recognised.

**Extraction quality is the bottleneck of the full system.** The policy is near-perfect given a
correct state, so the system-level difference between the two extractors (0.893 vs 0.990
satisfaction) is entirely the extraction difference. Hand-written rules reach the LLM on the
data they were tuned on and fall behind on new phrasings; the LLM's errors on held-out data are
small in number and mostly single-turn (a missed exclusion, a reset read as a removal).

**Grounding by construction.** Generating explanations from tool output rather than asking a
model to write them yields zero unsupported claims at zero model cost. The claim verifier is
also a ready-made reviewer for any future model-written explanation.

## 4. Limitations and evaluation boundary

- The scenarios are scripted and simulated. They test task completion under fixed user
  behaviour; they do not measure real-user satisfaction, long-term effects, or how users react
  to clarifications and infeasibility reports (spec: evaluation boundary).
- Scenario annotation follows a fixed mapping (for example "dinner" means main dish, "quick"
  is an objective, not a constraint). A different convention would change absolute numbers,
  though not the comparison between systems, which share the annotations.
- The LLM prompt and the rule parser were tuned on the same 30 development scenarios by the
  same author; the held-out split is the only unbiased comparison, and it is modest in size
  (267 turns, 19 infeasible, 12 clarify). Differences of one or two turns in the small
  categories are not significant.
- Ingredient matching is lexical over ingredient lines, with a small synonym and category
  layer; it will miss unusual ingredient names and treats "chicken broth" as containing
  chicken for exclusions (deliberately) but not for inclusions.
- Ranking inside the feasible set is by objective, relevance and popularity. The agent
  recommends what satisfies the constraints, not necessarily what a user would most enjoy;
  preference-based ranking was out of scope.
- The LLM is served non-deterministically: the same prompt gave one differing persistent
  flag between Groq and Lehmus on development data. All model outputs are cached with the
  results so the reported numbers are reproducible from the cache.

## 5. Optional extensions

All three optional extensions from the specification were implemented after the main
evaluation, as additions that leave the core system and its numbers unchanged.

### 5.1 Reviewer agent: model-written explanations checked against tool output

In the main system the explanation is assembled from tool output, which makes "unsupported
claims" zero by construction. The extension lets the language model write the explanation
instead, from the same evidence as structured data (`explain="llm"`, prompt in
`src/explain.py`), and adds a **reviewer** that re-checks the text: every quantity with a unit
("25 min", "420 kcal", "31 g protein", "9 recipes") is attributed to the recipe named just before
it and compared with the recipe's real value (tolerating rounding), the user's own requirement
values, or the infeasibility diagnosis; every "meets all your requirements" statement is
re-checked with the constraint checker. With `review=True`, an unsupported claim makes the agent
fall back to the templated answer. The reviewer is deterministic code and costs no model calls.

| Split (full agent, LLM extractor) | Turns with a model-written explanation | Claims checked | Unsupported | Turns corrected |
|---|---|---|---|---|
| dev (30 scenarios) | 104 | 630 | 0 | 0 |
| held-out (70 scenarios) | 258 | 1,575 | 0 | 0 |

Two findings. First, given the evidence as structured data, gpt-oss-120b did not invent or
alter a single number: all 2,205 quantities and "meets all requirements" statements it wrote
across both splits matched the tool output. The model-written answers are also more natural
than the templates, for example "For the lowest-fat goal the salad is best with 0.2 g fat, while
the highest-protein goal is met by Spinach Pie with 94.7 g protein." Second, the reviewer itself
needed two rounds of correction before it was trustworthy. Its first pass flagged 13 claims in
5 development turns and 4 claims in 2 held-out turns, every one a false alarm: the model's
typography (narrow no-break spaces inside numbers, curly quotes and non-breaking hyphens inside
recipe titles, markdown emphasis, JSON-escaped quotes echoed from the evidence) broke the title
matching, so numbers were attributed to the wrong recipe; a number inside a recipe title ("Fish
Tacos, 20 Minutes Max") was read as a claim; and a sentence that referred back to an earlier
recipe ("the salad is best with 0.2 g fat") was attributed to the recipe named last. After
normalising typography, ignoring numbers inside titles, and accepting a value that belongs to
another listed recipe (recorded as such, so the attribution stays auditable), the reviewer agreed
with a manual reading of every flagged case. The lesson is the same as for the main system: a
checker must be validated against the data it checks, and every flag it raised had to be read by
a person before the zero could be believed.

Cost: one extra model call per recommend or infeasible turn, about 1,500 tokens each with
reasoning effort low; 258 calls and 0.4 M tokens for the held-out split. The deterministic
reviewer adds no model calls. Model outputs are cached, so re-scoring after a reviewer change
costs nothing.

### 5.2 Preference-based ranking

Within the set of recipes that satisfy every active constraint, the main system ranks by
objectives, then relevance, then popularity (rating × log ratings). That is correct but
indifferent to portion size and preparation time, so a "vegetarian dinner" could be answered
with a 1,727 kcal pizza dough. The extension (`ranking="preference"`, `preference_score` in
`src/tools.py`) replaces popularity with a transparent score: community rating damped for few
ratings (weight 0.5), a portion-sanity prior centred on 450 kcal per serving (0.3), and a
mild penalty for preparation times above an hour (0.2). Explicit objectives still come first.

| Held-out, full agent | Constraint satisfaction | Silent relaxations | Mean rating | Mean kcal | Mean minutes | Share of recommendations in 100–800 kcal |
|---|---|---|---|---|---|---|
| Rules extractor, popularity | 0.893 | 29 | 4.73 | 395 | 47.6 | 0.80 |
| Rules extractor, preference | 0.896 | 29 | 4.73 | 401 | 34.2 | 0.95 |
| LLM extractor, popularity | 0.990 | 3 | 4.72 | 390 | 49.8 | 0.81 |
| LLM extractor, preference | 0.987 | 3 | 4.73 | 402 | 36.7 | 0.95 |

Constraint satisfaction and relaxations are unchanged within noise, as they must be: only the
order inside the feasible set changes. The recommendations become faster (34 to 37 vs 48 to 50
minutes on average) and far more often normally portioned (95 % vs 80 % between 100 and 800
kcal) at the same mean rating. There is no gold standard for "preferred", so this is a proxy evaluation; the
demo page has a toggle for a qualitative comparison.

### 5.3 Longer conversations

Ten additional scripted dialogues of 8 to 12 turns (101 turns, `scenarios/long`), annotated and
validated like the others and used only for evaluation, test whether memory degrades as
dialogues grow. They combine every phenomenon: lasting preferences, several replacements and
removals, infeasible requests with relaxation, trade-offs, and up to two resets per dialogue.

| Extractor | Extraction F1 | State exact | State exact by turn position (turns 1–10) |
|---|---|---|---|
| Rules | 0.961 | 0.753 | 0.90 · 0.90 · 0.90 · 0.90 · 0.80 · 0.70 · 0.60 · 0.50 · 0.60 · 0.57 |
| LLM | 0.996 | 0.970 | 1.00 · 1.00 · 1.00 · 1.00 · 1.00 · 1.00 · 1.00 · 1.00 · 0.90 · 0.86 |
| Rules, no memory | 0.948 | 0.109 | |
| LLM, no memory | 0.979 | 0.119 | |

| Long set, LLM extractor | Constraint satisfaction | Silent relaxations | Infeasible recall |
|---|---|---|---|
| RAG baseline | 0.181 | 93 | 0.00 |
| RAG + memory | 0.949 | 5 | 0.00 |
| Full agent | 1.000 | 0 | 1.00 |
| Agent, no memory | 0.237 | 69 | 0.00 |
| Agent, no validation | 0.214 | 90 | 0.00 |

The rule-based reader degrades steadily with dialogue length: one missed update early in a
dialogue stays in the state, so by turn 8 only half the states are exact. The model-based reader
holds a perfect state through eight turns and slips only in the last turns of the longest
dialogues. Without memory, state accuracy on long dialogues collapses to about 0.11 for both
readers, and the agent's constraint satisfaction with it (0.24). The full agent with the LLM
reader satisfies every constraint on all 101 turns.

## 6. Reproduction

```
python -m src.data_prep                                   # recipes.jsonl from the HUMMUS csv
python -m src.scenarios dev && python -m src.scenarios test   # validate annotations
python -m pytest -q                                       # 72 tests
python -m src.evaluate --split test --extractor rules     # extraction + state metrics
python -m src.evaluate --split test --extractor llm --provider lehmus
python -m src.run_agent --split test --mode agent [--extractor llm --provider lehmus]
python -m src.run_agent --split test --mode rag | rag_memory | agent --no-memory | agent --no-validate
python -m src.run_agent --split test --mode agent --ranking preference                      # extension 5.2
python -m src.run_agent --split test --mode agent --extractor llm --provider lehmus --explain llm --review   # 5.1
python -m src.evaluate --split long --extractor llm --provider lehmus                        # extension 5.3
```

Results are in `results/<split>_<config>[-<extractor>].json`, per-turn conversation logs in
`logs/<split>_<config>[-<extractor>].jsonl`, cached model outputs in `.cache/llm/`.
