# recipe-agent

A conversational recipe assistant that extracts user requirements turn by turn, keeps them in
a structured dialogue state, checks every recommendation with deterministic tools, and
explains its choices with evidence from the recipe data. Built for NLP course Project 25
(University of Oulu, autumn 2025).

Research question: do structured memory and explicit tool use improve constraint satisfaction
as user requirements change during a conversation? Short answer from the held-out evaluation:
yes. See [docs/evaluation_report.md](docs/evaluation_report.md).

| System (held-out, 70 dialogues) | Constraint satisfaction | Silent relaxations |
|---|---|---|
| Conversational RAG baseline | 0.25 | 237 |
| RAG + structured memory | 0.91 | 24 |
| Full tool-using agent | 0.99 | 3 |

## Setup

```
python -m venv .venv && .venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env                                  # fill in the key(s) you use
```

The recipe database `data/processed/recipes.jsonl` (5,000 recipes) is in the repository. To
rebuild it, put the HUMMUS `pp_recipes.csv` under `data/pp_recipes/` and run
`python -m src.data_prep` (see [docs/data_notes.md](docs/data_notes.md)).

LLM providers for the extractor (`--provider`): `lehmus` (Lehmus AI, University of Oulu,
OpenAI-compatible; used for the final evaluation), `groq` (free tier; used for the pilot),
`anthropic` (Claude). The rule-based extractor needs no key.

## Running

```
python -m pytest -q                                   # 72 tests, ~30 s
python -m src.scenarios dev                           # validate scenarios, show feasibility per turn
python -m src.evaluate --split dev --extractor rules  # extraction F1 + dialogue-state accuracy
python -m src.evaluate --split dev --extractor llm --provider lehmus [--fresh] [--scenarios 10]
python -m src.run_agent --split dev --mode agent      # full agent: actions, satisfaction, claims, tools
python -m src.run_agent --split dev --mode rag | rag_memory
python -m src.run_agent --split dev --mode agent --no-memory | --no-validate   # ablations
```

`run_agent` writes a per-turn conversation log to `logs/` and metrics to `results/`;
`--errors` prints the turns with a wrong action or a violated constraint. LLM responses are
cached in `.cache/llm/` keyed by provider, model, prompt version and input, so re-runs are
free; `--fresh` bypasses the cache to measure real usage.

Try the agent in the browser (chat plus live dialogue state, decision, tool calls and grounding
check; switch between the five system configurations; replay dev scenarios turn by turn):

```
python -m src.demo                                    # http://127.0.0.1:7860, rule-based extractor
python -m src.demo --extractor llm --provider lehmus
```

Or from Python:

```python
from src.agent import Agent
from src.tools import RecipeDB
agent = Agent(RecipeDB.load())           # rule-based extractor by default
for turn in ["Suggest a vegetarian dinner.", "Under 25 minutes.", "Avoid mushrooms."]:
    print(agent.turn(turn)["message"])
```

## Layout

| Path | What |
|---|---|
| `src/data_prep.py` | HUMMUS exploration, filters, diet flags, ingredient categories, 5k sample |
| `src/schema.py` | Constraint / objective / turn-extraction schema |
| `src/extract_rules.py` | Rule-based extractor |
| `src/extract_llm.py` | LLM extractor (Lehmus / Groq / Anthropic), schema validation, cache, usage |
| `src/state.py` | Dialogue state: persistent vs temporary, replacement rules, provenance |
| `src/tools.py` | `search_recipes`, `get_recipe_details`, `validate_constraints`, `diagnose_constraints` |
| `src/agent.py` | Decision policy, baselines and ablation switches, grounded explanations, claim verifier |
| `src/evaluate.py` | Extraction and state metrics, memory ablation |
| `src/run_agent.py` | System-level metrics and conversation logs |
| `src/scenarios.py`, `src/author.py` | Scenario loading, validation, and the compact authoring format |
| `scenarios/` | 30 dev + 70 held-out annotated dialogues; `README.md` has the annotation rules |
| `docs/` | Data notes, pilot summary, evaluation report |
| `results/`, `logs/` | Metrics and per-turn logs for every run in the report |
| `tests/` | Unit tests for every module |
