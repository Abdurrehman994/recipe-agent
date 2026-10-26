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
import html
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
CONFIG_BLURB = {
    "Full agent": "Extracts constraints, keeps them in the dialogue state, filters retrieval by them, "
                  "validates every candidate with deterministic tools, reports infeasibility with a "
                  "diagnosis and never relaxes a constraint.",
    "RAG baseline (transcript retrieval, no state)": "Keyword retrieval over the raw transcript. No "
                  "structured state, no filtering, no validation: it always recommends.",
    "RAG + structured memory (no validation)": "The dialogue state filters retrieval, but nothing is "
                  "validated; when nothing matches it silently falls back to unfiltered retrieval.",
    "Agent without memory (ablation)": "Full agent, but the state is rebuilt from the current turn only: "
                  "earlier requirements are forgotten.",
    "Agent without validation (ablation)": "Full agent, but constraints reach retrieval only as search "
                  "keywords; no hard filter and no validation, so infeasibility is never detected.",
}
EXAMPLE_TURNS = ["Suggest a vegetarian dinner.", "Under 25 minutes.", "Avoid mushrooms.",
                 "Actually, 40 minutes is fine.", "I'm allergic to nuts.",
                 "Forget that, show me a dessert.", "Which is the quickest and which has the most protein?",
                 "A vegan gluten-free dessert under 5 minutes with 40 g of protein.", "Something healthy."]
STATE_HEADERS = ["kind", "target", "value", "from turn", "persistent"]
TOOL_HEADERS = ["#", "tool", "arguments", "result"]
ACTION_LABEL = {"recommend": "Recommend", "clarify": "Clarify", "report_infeasible": "Infeasible"}

CSS = """
@keyframes rzFadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
@keyframes rzPop{0%{transform:scale(.6);opacity:0}70%{transform:scale(1.08);opacity:1}100%{transform:scale(1)}}
@keyframes rzShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes rzPulse{0%,100%{opacity:.35}50%{opacity:1}}
@keyframes rzGlow{0%{box-shadow:0 0 0 0 rgba(43,93,122,.45)}100%{box-shadow:0 0 0 14px rgba(43,93,122,0)}}
.gradio-container{background:radial-gradient(1200px 500px at 10% -10%,#e6eef3 0%,transparent 60%),radial-gradient(900px 400px at 100% 0%,#f3ece2 0%,transparent 55%),#f7f6f2 !important}
.rz-header{position:relative;overflow:hidden;background:linear-gradient(120deg,#1b2630,#2b5d7a,#1f3c4f,#2b5d7a);background-size:300% 300%;animation:rzShift 18s ease infinite;color:#f6f4ee;border-radius:16px;padding:24px 30px;margin-bottom:8px;box-shadow:0 12px 30px rgba(31,42,51,.18)}
.rz-header:after{content:"";position:absolute;right:-60px;top:-80px;width:260px;height:260px;border-radius:50%;background:radial-gradient(circle,rgba(200,121,42,.35),transparent 65%)}
.rz-header h1{margin:0 0 4px;font-size:26px;font-weight:600;color:#f6f4ee;letter-spacing:.2px}
.rz-header p{margin:0;font-size:15px;color:#cfe0e6}
.rz-chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
.rz-chip{font-size:12.5px;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.25);padding:3px 10px;border-radius:999px;color:#f6f4ee}
.rz-tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.rz-tile{background:#fff;border:1px solid #e3e6ea;border-radius:12px;padding:12px 14px;box-shadow:0 2px 10px rgba(31,42,51,.05);animation:rzFadeUp .4s ease both;transition:transform .15s ease,box-shadow .15s ease}
.rz-tile:hover{transform:translateY(-2px);box-shadow:0 6px 18px rgba(31,42,51,.10)}
.rz-tile:nth-child(2){animation-delay:.06s}.rz-tile:nth-child(3){animation-delay:.12s}.rz-tile:nth-child(4){animation-delay:.18s}
.rz-tile .k{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--body-text-color-subdued)}
.rz-tile .v{font-size:22px;font-weight:600;margin-top:2px;line-height:1.2}
.rz-badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:14px;font-weight:600;color:#fff;animation:rzPop .45s cubic-bezier(.2,.8,.2,1) both,rzGlow .9s ease-out .1s}
.rz-thinking{display:inline-flex;gap:5px;align-items:center;color:var(--body-text-color-subdued)}
.rz-thinking i{width:7px;height:7px;border-radius:50%;background:#2b5d7a;display:inline-block;animation:rzPulse 1s ease-in-out infinite}
.rz-thinking i:nth-child(2){animation-delay:.2s}.rz-thinking i:nth-child(3){animation-delay:.4s}
.rz-recommend{background:#2f7d4f}.rz-clarify{background:#b8860b}.rz-infeasible{background:#b3261e}.rz-none{background:#8a94a0}
.rz-warn{background:#fdecea;border-left:4px solid #b3261e;padding:8px 12px;border-radius:8px;margin-top:6px;color:#1f2328;animation:rzFadeUp .35s ease both}
.rz-muted{color:var(--body-text-color-subdued)}
.rz-panel{border-radius:14px !important;box-shadow:0 2px 12px rgba(31,42,51,.06);animation:rzFadeUp .4s ease both}
.rz-panel table tr{animation:rzFadeUp .3s ease both}
.rz-decision{animation:rzFadeUp .35s ease both}
.rz-examples button{font-size:13px;border-radius:999px !important;transition:transform .15s ease,box-shadow .15s ease,background .15s ease}
.rz-examples button:hover{transform:translateY(-2px);box-shadow:0 6px 16px rgba(43,93,122,.18)}
.message-row,.bubble-wrap .message{animation:rzFadeUp .35s ease both}
button.primary{background:linear-gradient(135deg,#2b5d7a,#1f3c4f) !important;border:none !important;transition:transform .15s ease,box-shadow .15s ease}
button.primary:hover{transform:translateY(-1px);box-shadow:0 8px 18px rgba(43,93,122,.3)}
@media (prefers-reduced-motion:reduce){*{animation:none !important;transition:none !important}}
footer{display:none !important}
"""


def _short(v, n: int = 150) -> str:
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 1] + "…"


class Session:
    """The one live dialogue behind the page."""

    def __init__(self, db: RecipeDB, extractor):
        self.db, self.extractor = db, extractor
        self.lock = threading.Lock()
        self.scenarios = {sc["id"]: sc["turns"] for sc in load_scenarios("dev")}
        self.scenario, self.pos = None, 0
        self.config = "Full agent"
        self.new_dialogue(self.config)

    def new_dialogue(self, config: str) -> None:
        with self.lock:
            self.config = config
            self.agent = Agent(self.db, self.extractor, **CONFIGS[config])
            self.pos = 0

    def turn(self, text: str) -> dict:
        with self.lock:
            rec = self.agent.turn(text)
        rec["unsupported"] = verify_claims(self.db, rec["claims"])
        return rec


SESSION: Session | None = None


# ------------------------------------------------------------------ view helpers
def header_html() -> str:
    ex = getattr(SESSION.extractor, "name", "rules")
    return (f'<div class="rz-header"><h1>Conversational Recipe Agent</h1>'
            f'<p>Memory, deterministic tools and grounded explanations · NLP Project 25 · University of Oulu</p>'
            f'<div class="rz-chips"><span class="rz-chip">Extractor: {html.escape(ex)}</span>'
            f'<span class="rz-chip">{len(SESSION.db.ids):,} recipes (HUMMUS)</span>'
            f'<span class="rz-chip">{len(SESSION.scenarios)} scripted dev scenarios</span></div></div>')


def tiles_html(rec: dict | None) -> str:
    if rec is None:
        return ('<div class="rz-tiles">' + "".join(
            f'<div class="rz-tile"><div class="k">{k}</div><div class="v rz-muted">—</div></div>'
            for k in ("Turn", "Action", "Tool calls", "Claims checked")) + "</div>")
    action = rec["action"]
    badge = f'<span class="rz-badge rz-{action.split("_")[-1]}">{ACTION_LABEL[action]}</span>'
    n, bad = len(rec["claims"]), len(rec["unsupported"])
    claims = f"{n}" + (f' <span style="color:#b3261e;font-size:14px">{bad} unsupported</span>' if bad else
                       (' <span style="color:#2f7d4f;font-size:14px">all supported</span>' if n else ""))
    return (f'<div class="rz-tiles">'
            f'<div class="rz-tile"><div class="k">Turn</div><div class="v">{rec["turn"]}</div></div>'
            f'<div class="rz-tile"><div class="k">Action</div><div class="v">{badge}</div></div>'
            f'<div class="rz-tile"><div class="k">Tool calls</div><div class="v">{len(rec["tool_calls"])}</div></div>'
            f'<div class="rz-tile"><div class="k">Claims checked</div><div class="v">{claims}</div></div></div>')


def state_rows(rec: dict) -> list[list]:
    return [[a["kind"], a["target"], "" if a.get("value") is None else f"{a['value']:g}",
             a["source_turn"], "yes" if a["persistent"] else ""] for a in rec["state"]]


def state_note(rec: dict) -> str:
    parts = []
    if rec.get("objectives"):
        parts.append("**Objectives:** " + ", ".join(
            f"{'lowest' if o['direction'] == 'min' else 'highest'} {o['target']}" for o in rec["objectives"]))
    if rec.get("topic"):
        parts.append("**Retrieval topic:** " + ", ".join(rec["topic"]))
    if not rec["state"]:
        parts.append("No active constraints." if SESSION.agent.mode != "rag"
                     else "This configuration keeps no structured state.")
    return "  \n".join(parts)


def decision_html(rec: dict) -> str:
    out = []
    if rec.get("relaxed"):
        out.append('<div class="rz-warn"><b>Silent relaxation.</b> Nothing matched the constraints, so this '
                   'configuration fell back to unfiltered retrieval and recommended anyway.</div>')
    if rec["recommendations"]:
        items = "".join(f"<li>{html.escape(r['title'])}"
                        + (' <span style="color:#2f7d4f">· verified</span>' if r["verified"]
                           else ' <span class="rz-muted">· not validated</span>') + "</li>"
                        for r in rec["recommendations"])
        out.append(f"<b>Recommended</b><ul style='margin:4px 0 0 18px'>{items}</ul>")
    if rec.get("rejected"):
        items = "".join(f"<li>{html.escape(r['title'])} <span class='rz-muted'>failed: "
                        f"{html.escape(', '.join(r['failed']))}</span></li>" for r in rec["rejected"])
        out.append(f"<b>Rejected after validation</b><ul style='margin:4px 0 0 18px'>{items}</ul>")
    if rec["action"] == "clarify":
        out.append("Nothing in the dialogue can be turned into a constraint yet, so the agent asks "
                   "instead of guessing. No tool was called.")
    if rec["action"] == "report_infeasible":
        out.append("No recipe satisfies every active constraint. The diagnosis names the most "
                   "restrictive one; the state keeps all of them.")
    return "<div>" + "<br>".join(out) + "</div>" if out else "<span class='rz-muted'>—</span>"


def tool_rows(rec: dict) -> list[list]:
    return [[i + 1, c["tool"], _short(c["args"]), _short(c["result"])]
            for i, c in enumerate(rec["tool_calls"])]


def claims_md(rec: dict) -> str:
    n, bad = len(rec["claims"]), rec["unsupported"]
    if not n:
        return "No factual claims in this turn."
    if not bad:
        return f"**{n} factual claims** in the answer were re-checked against the database: all supported."
    return (f"**{n} factual claims** re-checked: **{len(bad)} unsupported**\n\n```json\n"
            + json.dumps(bad, indent=1, ensure_ascii=False) + "\n```")


def chat_message(rec: dict) -> str:
    return f"**{ACTION_LABEL[rec['action']]}** · turn {rec['turn']}\n\n{rec['message']}"


# ------------------------------------------------------------------ event handlers
N_OUT = 7  # chat, tiles, state, state_note, decision, tools, claims (+ the cleared textbox)


THINKING = ('<span class="rz-thinking"><i></i><i></i><i></i> extracting constraints, searching, '
            'validating</span>')


def respond(text: str, history: list[dict]):
    """Generator: first the user turn with a thinking indicator, then the full result."""
    text = (text or "").strip()
    if not text:
        yield (history, *([gr.update()] * (N_OUT - 1)), "")
        return
    history = (history or []) + [{"role": "user", "content": text}]
    yield (history + [{"role": "assistant", "content": THINKING}],
           *([gr.update()] * (N_OUT - 1)), "")
    rec = SESSION.turn(text)
    history = history + [{"role": "assistant", "content": chat_message(rec)}]
    yield (history, tiles_html(rec), state_rows(rec), state_note(rec), decision_html(rec),
           tool_rows(rec), claims_md(rec), "")


def make_sender(text: str):
    """Click handler sending a fixed example turn (a real generator function for Gradio)."""
    def send(history: list[dict]):
        yield from respond(text, history)
    return send


def new_dialogue(config: str):
    SESSION.new_dialogue(config)
    return ([], tiles_html(None), [], "No active constraints.", "<span class='rz-muted'>—</span>",
            [], "Every factual statement in an answer is re-checked against the recipe database.", "")


def describe_config(config: str) -> str:
    return f"**{config}.** {CONFIG_BLURB[config]}"


def load_scenario(choice: str):
    SESSION.scenario = SESSION.scenarios.get(choice)
    SESSION.pos = 0
    if not SESSION.scenario:
        return "Pick a scenario to replay it turn by turn."
    return f"**{choice}** loaded: {len(SESSION.scenario)} scripted turns. Press *Send next turn*."


def next_turn(history: list[dict]):
    sc = SESSION.scenario
    if not sc or SESSION.pos >= len(sc):
        yield (history, *([gr.update()] * (N_OUT - 1)), "",
               "Scenario finished. Load another or type your own turns.")
        return
    t = sc[SESSION.pos]
    SESSION.pos += 1
    note = (f"Turn {SESSION.pos}/{len(sc)} sent. Annotated expected action: "
            f"**{ACTION_LABEL[t['expected_action']]}**.")
    for out in respond(t["user"], history):
        yield (*out, note)


# ------------------------------------------------------------------ page
def build() -> gr.Blocks:
    with gr.Blocks(title="Recipe agent demo", fill_width=True) as page:
        gr.HTML(header_html())
        with gr.Row(equal_height=True):
            config = gr.Dropdown(list(CONFIGS), value="Full agent", label="System configuration", scale=5)
            reset_btn = gr.Button("New dialogue", variant="secondary", scale=1)
            scenario = gr.Dropdown(["—"] + list(SESSION.scenarios), value="—",
                                   label="Replay a scripted dev scenario", scale=4)
            next_btn = gr.Button("Send next turn", variant="primary", scale=1)
        with gr.Row():
            config_md = gr.Markdown(describe_config("Full agent"), elem_classes=["rz-muted"])
            note = gr.Markdown("Pick a scenario to replay it turn by turn.", elem_classes=["rz-muted"])

        with gr.Row():
            with gr.Column(scale=11):
                chat = gr.Chatbot(label="Conversation", height=560, layout="bubble",
                                  placeholder="Start a dialogue below, or replay a scripted scenario.")
                box = gr.Textbox(placeholder="Type a turn, e.g. Suggest a vegetarian dinner.",
                                 label="Your turn", autofocus=True, submit_btn="Send")
                gr.Markdown("Example turns (click to send):", elem_classes=["rz-muted"])
                with gr.Row(elem_classes=["rz-examples"]):
                    example_btns = [gr.Button(t, size="sm", variant="secondary") for t in EXAMPLE_TURNS]
            with gr.Column(scale=9):
                tiles = gr.HTML(tiles_html(None))
                with gr.Group(elem_classes=["rz-panel"]):
                    state = gr.Dataframe(headers=STATE_HEADERS, value=[], label="Dialogue state (memory)",
                                         interactive=False, wrap=True,
                                         column_widths=["18%", "32%", "14%", "16%", "20%"])
                    state_md = gr.Markdown("No active constraints.", elem_classes=["rz-muted"])
                with gr.Group(elem_classes=["rz-panel"]):
                    gr.Markdown("**Decision**")
                    decision = gr.HTML("<span class='rz-muted'>—</span>", elem_classes=["rz-decision"])
                with gr.Group(elem_classes=["rz-panel"]):
                    tools = gr.Dataframe(headers=TOOL_HEADERS, value=[], label="Tool calls this turn",
                                         interactive=False, wrap=True, max_height=260,
                                         column_widths=["6%", "22%", "36%", "36%"])
                with gr.Group(elem_classes=["rz-panel"]):
                    gr.Markdown("**Grounding check**")
                    claims = gr.Markdown("Every factual statement in an answer is re-checked against the "
                                         "recipe database.", elem_classes=["rz-muted"])
                with gr.Accordion("How to read this page", open=False):
                    gr.Markdown(
                        "- **Dialogue state**: the constraints currently in force, the turn that set "
                        "each one, and whether it is a standing preference (persistent).\n"
                        "- **Action**: Recommend, Clarify (nothing to constrain yet) or Infeasible "
                        "(nothing matches; the agent names the blocking requirement and drops nothing).\n"
                        "- **Tool calls**: search, details, validation and diagnosis are deterministic code; "
                        "the model only converts text into constraints.\n"
                        "- **Grounding check**: every number and every “meets requirement” statement in the "
                        "answer is re-verified against the database.")

        outputs = [chat, tiles, state, state_md, decision, tools, claims, box]
        box.submit(respond, [box, chat], outputs)
        for btn, text in zip(example_btns, EXAMPLE_TURNS):
            btn.click(make_sender(text), [chat], outputs)
        reset_btn.click(new_dialogue, [config], outputs)
        config.change(new_dialogue, [config], outputs)
        config.change(describe_config, [config], [config_md])
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
    theme = gr.themes.Default(primary_hue="blue", neutral_hue="slate",
                              font=[gr.themes.GoogleFont("IBM Plex Sans"), "Segoe UI", "sans-serif"])
    build().launch(server_name="127.0.0.1", server_port=args.port, inbrowser=not args.no_browser,
                   theme=theme, css=CSS)


if __name__ == "__main__":
    main()
