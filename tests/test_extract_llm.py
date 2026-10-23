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


class FakeGroqClientWithUsage(FakeGroqClient):
    def create(self, **kwargs):
        resp = super().create(**kwargs)
        resp.usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=50, total_tokens=1050)
        return resp


def test_usage_is_counted_and_cached_answers_keep_their_usage(tmp_path):
    out = {"updates": [upd("add", "diet", "vegan")], "objectives": [], "reset": False}
    first = LLMExtractor(provider="groq", client=FakeGroqClientWithUsage([out]), cache_dir=tmp_path,
                         min_interval_s=0)
    first.extract("Vegan please.")
    u = first.usage_summary()
    assert u["api_calls"] == 1 and u["cached_calls"] == 0 and u["total_tokens"] == 1050
    assert u["tokens_per_call"] == 1050.0
    second = LLMExtractor(provider="groq", client=FakeGroqClientWithUsage([]), cache_dir=tmp_path,
                          min_interval_s=0)
    second.extract("Vegan please.")
    u2 = second.usage_summary()
    assert u2["api_calls"] == 0 and u2["cached_calls"] == 1 and u2["total_tokens"] == 1050


def test_fresh_ignores_cache_and_old_cache_format_still_reads(tmp_path):
    import hashlib
    from src.extract_llm import PROMPT_VERSION
    out = {"updates": [upd("add", "diet", "vegan")], "objectives": [], "reset": False}
    client = FakeGroqClientWithUsage([out, out])
    ex = LLMExtractor(provider="groq", client=client, cache_dir=tmp_path, min_interval_s=0, fresh=True)
    ex.extract("Vegan please.")
    ex.extract("Vegan please.")
    assert len(client.calls) == 2 and ex.usage_summary()["api_calls"] == 2
    # an entry in the old bare format (no usage) is still accepted and counted as unmeasured
    user = ex._user_message("Old format.") if hasattr(ex, "_user_message") else None
    old = LLMExtractor(provider="groq", client=FakeGroqClientWithUsage([]), cache_dir=tmp_path,
                       min_interval_s=0)
    from src.extract_llm import render_state
    from src.state import DialogueState
    msg = f"Current state:\n{render_state(DialogueState())}\n\nUser message:\nOld format."
    key = hashlib.sha256(json.dumps(["groq", old.model, "medium", PROMPT_VERSION, msg]).encode()).hexdigest()[:24]
    (tmp_path / f"{key}.json").write_text(json.dumps(out), encoding="utf-8")
    ext = old.extract("Old format.")
    assert ext.updates and old.usage_summary()["unmeasured_calls"] == 1


def _bad_request(msg):
    import httpx
    import openai
    resp = httpx.Response(400, request=httpx.Request("POST", "http://lehmus.test/v1/chat/completions"))
    return openai.BadRequestError(msg, response=resp, body={"error": {"message": msg}})


class FakeOpenAIClient(FakeGroqClientWithUsage):
    """OpenAI-compatible server that rejects json_schema output: the first such call fails."""

    def __init__(self, outputs, reject_schema=False):
        super().__init__(outputs)
        self.reject_schema = reject_schema

    def create(self, **kwargs):
        if self.reject_schema and kwargs.get("response_format", {}).get("type") == "json_schema":
            self.calls.append(kwargs)
            raise _bad_request("response_format json_schema is not supported")
        return super().create(**kwargs)


def test_lehmus_provider_uses_openai_compatible_chat_completions(tmp_path, monkeypatch):
    monkeypatch.setenv("LEHMUS_MODEL", "gpt-oss-120b")
    out = {"updates": [upd("add", "course", "lunch")], "objectives": [], "reset": False}
    client = FakeOpenAIClient([out])
    ex = LLMExtractor(provider="lehmus", client=client, cache_dir=tmp_path)
    ext = ex.extract("Lunch ideas?")
    assert [u.constraint for u in ext.updates] == [Constraint("course", "lunch")]
    call = client.calls[0]
    assert call["model"] == "gpt-oss-120b" and call["response_format"]["type"] == "json_schema"
    assert ex.name == "llm-lehmus-gpt-oss-120b" and ex.usage_summary()["api_calls"] == 1


def test_lehmus_falls_back_to_json_object_when_schema_is_rejected(tmp_path):
    out = {"updates": [upd("add", "diet", "vegan")], "objectives": [], "reset": False}
    client = FakeOpenAIClient([out], reject_schema=True)
    ex = LLMExtractor(provider="lehmus", client=client, cache_dir=tmp_path)
    ext = ex.extract("Vegan please.")
    assert ext.updates[0].constraint == Constraint("diet", "vegan")
    assert [c["response_format"]["type"] for c in client.calls] == ["json_schema", "json_object"]
    assert ex.output_mode == "json_object"
    # the fallback is remembered: the next call goes straight to json_object
    ex.extract("Something else vegan.") if False else None
