"""Local Gradio demo: chat with the agent and watch its dialogue state, decision, tool calls
and grounding check next to every answer.

    python -m src.demo                                  # rule-based extractor
    python -m src.demo --extractor llm --provider lehmus
    python -m src.demo --port 7860 --no-browser

Opens http://127.0.0.1:7860 in the browser. The system configuration (RAG baseline, RAG +
structured memory, full agent, the two ablations) can be switched from the page, which starts a
new dialogue. Development scenarios can be loaded and sent turn by turn. One dialogue at a time
(a single presenter), which keeps the state display unambiguous.
"""
from __future__ import annotations

import argparse
import json
import threading

import gradio as gr

from src.agent import Agent, verify_claims
from src.scenarios import load_scenarios
from src.tools import RecipeDB

CONFIGS = {
    "Full agent": dict(mode="agent", memory=True, validate=True),
    "RAG baseline (transcript retrieval, no state)": dict(mode="rag", memory=True, validate=True),
    "RAG + structured memory (no validation)": dict(mode="rag_memory", memory=True, validate=True),
    "Agent without memory (ablation)": dict(mode="agent", memory=False, validate=True),
    "Agent without validation (ablation)": dict(mode="agent", memory=True, validate=False),
}
STATE_HEADERS = ["kind", "target", "value", "from turn", "persistent"]
TOOL_HEADERS = ["#", "tool", "arguments", "result"]


def _short(v, n: int = 140) -> str:
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 1] + "…"


class Session:
    """The one live dialogue behind the page."""

    def __init__(self, db: RecipeDB, extractor):
        self.db, self.extractor = db, extractor
        self.lock = threading.Lock()
        self.scenarios = {sc["id"]: sc["turns"] for sc in load_scenarios("dev")}
        self.scenario, self.pos = None, 0
        self.new_dialogue("Full agent")

    def new_dialogue(self, config: str) -> None:
        with self.lock:
            self.agent = Agent(self.db, self.extractor, **CONFIGS[config])
            self.pos = 0

    def turn(self, text: str) -> dict:
        with self.lock:
            rec = self.agent.turn(text)
        rec["unsupported"] = verify_claims(self.db, rec["claims"])
        return rec


SESSION: Session | None = None


# ------------------------------------------------------------------ view helpers
def state_rows(rec: dict) -> list[list]:
    return [[a["kind"], a["target"], "" if a.get("value") is None else f"{a['value']:g}",
             a["source_turn"], "yes" if a["persistent"] else ""] for a in rec["state"]]


def decision_md(rec: dict) -> str:
    action = rec["action"].replace("_", " ")
    lines = [f"**Turn {rec['turn']}: {action}**"]
    if rec.get("objectives"):
        lines.append("Objectives: " + ", ".join(
            f"{'lowest' if o['direction'] == 'min' else 'highest'} {o['target']}" for o in rec["objectives"]))
    if rec.get("topic"):
        lines.append("Retrieval topic: " + ", ".join(rec["topic"]))
    if rec.get("relaxed"):
        lines.append("**Warning:** fell back to unfiltered retrieval, a silent relaxation of the user's constraints.")
    if rec["recommendations"]:
        lines.append("Recommended: " + "; ".join(
            r["title"] + (" (verified)" if r["verified"] else "") for r in rec["recommendations"]))
    if rec.get("rejected"):
        lines.append("Rejected after validation: " + "; ".join(
            f"{r['title']} ({', '.join(r['failed'])})" for r in rec["rejected"]))
    return "\n\n".join(lines)


def tool_rows(rec: dict) -> list[list]:
    return [[i + 1, c["tool"], _short(c["args"]), _short(c["result"])]
            for i, c in enumerate(rec["tool_calls"])]


def claims_md(rec: dict) -> str:
    n, bad = len(rec["claims"]), rec["unsupported"]
    if not n:
        return "No factual claims in this turn."
    verdict = "all supported by the database" if not bad else f"**{len(bad)} unsupported**"
    text = f"{n} factual claims in the answer re-checked: {verdict}."
    if bad:
        text += "\n\n```json\n" + json.dumps(bad, indent=1, ensure_ascii=False) + "\n```"
    return text


# ------------------------------------------------------------------ event handlers
def respond(text: str, history: list[dict]):
    text = (text or "").strip()
    if not text:
        return history, gr.update(), gr.update(), gr.update(), gr.update(), ""
    rec = SESSION.turn(text)
    tag = {"recommend": "RECOMMEND", "clarify": "CLARIFY", "report_infeasible": "INFEASIBLE"}[rec["action"]]
    history = (history or []) + [{"role": "user", "content": text},
                                 {"role": "assistant", "content": f"[{tag}] {rec['message']}"}]
    return (history, state_rows(rec), decision_md(rec), tool_rows(rec), claims_md(rec), "")


def new_dialogue(config: str):
    SESSION.new_dialogue(config)
    return [], [], f"New dialogue with **{config}**. Extractor: {SESSION.agent.extractor.name}.", [], "", ""


def load_scenario(choice: str):
    SESSION.scenario = SESSION.scenarios.get(choice)
    SESSION.pos = 0
    if not SESSION.scenario:
        return "No scenario loaded."
    return f"Loaded {choice}: {len(SESSION.scenario)} scripted turns. Press *Send next turn*."


def next_turn(history: list[dict]):
    sc = SESSION.scenario
    if not sc or SESSION.pos >= len(sc):
        return (history, gr.update(), gr.update(), gr.update(), gr.update(), "",
                "Scenario finished. Load another or type your own turns.")
    t = sc[SESSION.pos]
    SESSION.pos += 1
    out = respond(t["user"], history)
    note = (f"Turn {SESSION.pos}/{len(sc)} sent. Annotated expected action: "
            f"**{t['expected_action'].replace('_', ' ')}**.")
    return (*out, note)


# ------------------------------------------------------------------ page
def build() -> gr.Blocks:
    with gr.Blocks(title="Recipe agent demo") as page:
        gr.Markdown("## Conversational recipe agent: memory, tools and grounded answers")
        with gr.Row():
            config = gr.Dropdown(list(CONFIGS), value="Full agent", label="System configuration",
                                 scale=3)
            reset_btn = gr.Button("New dialogue", scale=1)
            gr.Markdown(f"Extractor: **{SESSION.agent.extractor.name}**")
        with gr.Row():
            with gr.Column(scale=11):
                chat = gr.Chatbot(label="Conversation", height=520)
                box = gr.Textbox(placeholder="Type a turn, e.g. Suggest a vegetarian dinner.",
                                 label="Your turn", autofocus=True)
                with gr.Row():
                    scenario = gr.Dropdown(["—"] + list(SESSION.scenarios), value="—",
                                           label="Scripted development scenario", scale=3)
                    next_btn = gr.Button("Send next turn", scale=1)
                note = gr.Markdown("")
            with gr.Column(scale=9):
                state = gr.Dataframe(headers=STATE_HEADERS, value=[], label="Dialogue state (memory)",
                                     interactive=False, wrap=True)
                decision = gr.Markdown("No turn yet.", label="Decision")
                tools = gr.Dataframe(headers=TOOL_HEADERS, value=[], label="Tool calls this turn",
                                     interactive=False, wrap=True)
                claims = gr.Markdown("", label="Grounding check")

        outputs = [chat, state, decision, tools, claims, box]
        box.submit(respond, [box, chat], outputs)
        reset_btn.click(new_dialogue, [config], outputs)
        config.change(new_dialogue, [config], outputs)
        scenario.change(load_scenario, [scenario], [note])
        next_btn.click(next_turn, [chat], outputs + [note])
    return page


def main():
    global SESSION
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--extractor", default="rules", help="rules | llm")
    ap.add_argument("--provider", default="lehmus")
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    from src.evaluate import get_extractor
    SESSION = Session(RecipeDB.load(), get_extractor(args.extractor, args.provider, args.model, args.effort))
    build().launch(server_name="127.0.0.1", server_port=args.port, inbrowser=not args.no_browser)


if __name__ == "__main__":
    main()
