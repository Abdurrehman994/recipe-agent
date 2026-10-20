"""LLM extractor tests with a fake client: no network, no API key."""
import json
from types import SimpleNamespace

from src.extract_llm import LLMExtractor, parse_output, render_state
from src.schema import Constraint, ConstraintUpdate, Objective, TurnExtraction
from src.state import DialogueState


class FakeClient:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = json.dumps(self.outputs.pop(0))
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


def upd(op, kind, target, value=None, persistent=False):
    return {"op": op, "kind": kind, "target": target, "value": value, "persistent": persistent}


def test_parse_normalizes_ingredients_and_drops_invalid_items():
    data = {"updates": [upd("add", "exclude", "Mushrooms"), upd("add", "exclude", "seafood"),
                        upd("add", "diet", "keto"), upd("add", "max", "duration_min", 25)],
            "objectives": [{"target": "protein_g", "direction": "max"}], "reset": False}
    ext, dropped = parse_output(data)
    assert [u.constraint for u in ext.updates] == [
        Constraint("exclude", "mushroom"), Constraint("exclude", "shellfish"),
        Constraint("max", "duration_min", 25)]
    assert ext.objectives == [Objective("protein_g", "max")]
    assert len(dropped) == 1 and "keto" in dropped[0]


def test_remove_numeric_without_value_takes_value_from_state():
    state = DialogueState()
    state.apply(TurnExtraction([ConstraintUpdate("add", Constraint("max", "calories_kcal", 600))]), 1)
    ext, dropped = parse_output({"updates": [upd("remove", "max", "calories_kcal")],
                                 "objectives": [], "reset": False}, state)
    assert not dropped and ext.updates[0].op == "remove"
    state.apply(ext, 2)
    assert state.active == []


def test_state_is_rendered_without_turn_numbers():
    state = DialogueState()
    state.apply(TurnExtraction([ConstraintUpdate("add", Constraint("diet", "vegan"), True)]), 1)
    assert json.loads(render_state(state))["active_constraints"] == [
        {"kind": "diet", "target": "vegan", "persistent": True}]


def test_extract_calls_model_once_then_uses_cache(tmp_path):
    out = {"updates": [upd("add", "diet", "nut_free", persistent=True)], "objectives": [], "reset": False}
    client = FakeClient([out])
    ex = LLMExtractor(client=client, cache_dir=tmp_path)
    first = ex.extract("I'm allergic to nuts.")
    second = LLMExtractor(client=FakeClient([]), cache_dir=tmp_path).extract("I'm allergic to nuts.")
    assert first.updates == second.updates
    assert first.updates[0].persistent and len(client.calls) == 1
    assert client.calls[0]["output_config"]["format"]["type"] == "json_schema"


class FakeGroqClient:
    """Mimics groq.Groq().chat.completions.create."""

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = json.dumps(self.outputs.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=text))])


def test_groq_provider_uses_chat_completions_with_json_schema(tmp_path):
    out = {"updates": [upd("add", "exclude", "mushrooms"), upd("add", "max", "duration_min", 25)],
           "objectives": [], "reset": False}
    client = FakeGroqClient([out])
    ex = LLMExtractor(provider="groq", client=client, cache_dir=tmp_path, min_interval_s=0)
    ext = ex.extract("Under 25 minutes and no mushrooms.")
    assert [u.constraint for u in ext.updates] == [
        Constraint("exclude", "mushroom"), Constraint("max", "duration_min", 25)]
    call = client.calls[0]
    assert call["model"] == "openai/gpt-oss-120b"
    assert call["response_format"]["type"] == "json_schema"
    assert call["messages"][0]["role"] == "system" and call["reasoning_effort"] == "medium"
    assert ex.name == "llm-groq-gpt-oss-120b"


def test_provider_caches_are_separate(tmp_path):
    out = {"updates": [upd("add", "diet", "vegan")], "objectives": [], "reset": False}
    LLMExtractor(provider="groq", client=FakeGroqClient([out]), cache_dir=tmp_path,
                 min_interval_s=0).extract("Vegan please.")
    anthropic_client = FakeClient([out])
    LLMExtractor(client=anthropic_client, cache_dir=tmp_path).extract("Vegan please.")
    assert len(anthropic_client.calls) == 1  # groq's cached answer was not reused for Claude


def test_unknown_provider_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        LLMExtractor(provider="openai")
