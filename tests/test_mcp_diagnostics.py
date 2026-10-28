"""Opt-in MCP routing diagnostics: real core decisions and MCP parsing, no weights."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

from mcp.server.mcpserver.exceptions import ToolError as McpToolError  # noqa: E402
from laya import BaseHook, Router  # noqa: E402
from laya.mcp import server as server_mod  # noqa: E402
from laya.mcp.tools import ToolError, laya_predict, laya_route, laya_tournament  # noqa: E402


STATE = {"message": "Turn lights off"}
QUESTIONS = {"category": {"type": "choice", "instructions": "Classify the message.",
                          "criteria": {"A": "Support", "B": "Other"}}}
DIAGNOSTIC = {"stage": "whole_state", "reason": "insufficient_words", "source": None}
ROUTING = {"model": "english", "repo": "example/english", "reason": "test route"}
INVALID_FLAGS = [None, 0, 1, 0.0, 1.0, "false", "true", "", [], {}]


class LegacyRouter:
    """An injected router predating explain; even an unrequested keyword is incompatible."""

    def __init__(self):
        self.calls = []

    def route(self, state, questions):
        self.calls.append((state, questions))
        return dict(ROUTING)


class RecordingRouter(Router):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def route(self, state, questions=None, **kwargs):
        self.calls.append((state, questions, dict(kwargs)))
        return super().route(state, questions, **kwargs)


def _call_tool(arguments):
    result = asyncio.run(server_mod.server.call_tool("laya_route", arguments))
    assert not result.is_error
    return json.loads(result.content[0].text)


@pytest.mark.parametrize("flag", [{}, {"explain": False}])
def test_default_route_preserves_legacy_router_and_exact_output(flag):
    router = LegacyRouter()
    out = laya_route(STATE, QUESTIONS, router=router, **flag)
    assert json.dumps(out) == json.dumps(ROUTING)
    assert len(router.calls) == 1


@pytest.mark.parametrize(("state", "expected"), [
    (STATE, DIAGNOSTIC),
    ({"message": "xyzzy frobnitz quux blorb"},
     {"stage": "whole_state", "reason": "no_language_evidence", "source": None}),
    ({"note": "This is an English note about the order. " * 20,
      "payload": {"customer": "Ich kann mich nicht anmelden und brauche Hilfe"}},
     {"stage": "segment", "reason": "named_segment",
      "source": {"leaf_index": 1, "path": ["payload", "customer"], "line": 1}}),
])
def test_route_returns_the_core_diagnostic_without_detection_values(state, expected):
    router = RecordingRouter()
    original = copy.deepcopy(state)
    plain = laya_route(state, QUESTIONS, router=router)
    explained = laya_route(state, QUESTIONS, router=router, explain=True)
    assert explained.pop("diagnostic") == expected
    assert explained == plain
    assert set(explained) == {"model", "repo", "reason"}
    assert router.calls[0][2] == {}
    assert router.calls[1][2] == {"explain": True}
    assert len(router.calls) == 2
    assert state == original


@pytest.mark.parametrize("override", [
    {"model": "english"}, {"task": "typed_decisions"}, {"lang": "de"}, {"lang_guess": "de"},
])
def test_route_override_has_explicit_null_diagnostic(override):
    router = Router()
    plain = laya_route(STATE, QUESTIONS, router=router, **override)
    explained = laya_route(STATE, QUESTIONS, router=router, explain=True, **override)
    assert explained.pop("diagnostic") is None
    assert explained == plain


@pytest.mark.parametrize("replacement", [
    dict(ROUTING),
    dict(ROUTING, detection={"diagnostic": {"stage": "whole_state", "reason": "no_letters", "source": None}}),
])
def test_route_uses_final_hook_decision_not_stale_detection(replacement):
    class ReplaceRoute(BaseHook):
        def on_route(self, ctx):
            ctx.decision = copy.deepcopy(replacement)

    router = RecordingRouter(hooks=[ReplaceRoute()])
    out = laya_route(STATE, QUESTIONS, router=router, explain=True)
    assert out == dict(ROUTING, diagnostic=replacement.get("detection", {}).get("diagnostic"))
    assert len(router.calls) == 1


def test_object_style_route_decision_preserves_only_the_diagnostic():
    class ObjectRouter:
        def route(self, state, questions, *, explain=False):
            assert explain is True
            return SimpleNamespace(**ROUTING, detection={"mixed_segment": "private value",
                                                        "diagnostic": copy.deepcopy(DIAGNOSTIC)})

    out = laya_route(STATE, QUESTIONS, router=ObjectRouter(), explain=True)
    assert out == dict(ROUTING, diagnostic=DIAGNOSTIC)
    assert "private value" not in json.dumps(out)


@pytest.mark.parametrize("bad", INVALID_FLAGS)
def test_route_rejects_non_boolean_before_dispatch(bad):
    router = RecordingRouter()
    with pytest.raises(ToolError, match="explain") as info:
        laya_route(STATE, QUESTIONS, router=router, explain=bad)
    assert info.value.code == "invalid_explain"
    assert router.calls == []


def test_legacy_router_refuses_requested_explanation_without_fallback():
    router = LegacyRouter()
    with pytest.raises(ToolError, match="explain") as info:
        laya_route(STATE, QUESTIONS, router=router, explain=True)
    assert info.value.code == "unsupported_explain"
    assert router.calls == []


@pytest.mark.parametrize("message", [
    "internal routing failure",
    "wrapped.route() got an unexpected keyword argument 'explain'",
])
def test_internal_type_error_is_not_relabelled_as_unsupported(message):
    error = TypeError(message)

    class BrokenRouter:
        def route(self, state, questions, **kwargs):
            raise error

    with pytest.raises(TypeError) as info:
        laya_route(STATE, QUESTIONS, router=BrokenRouter(), explain=True)
    assert info.value is error


def test_forwarding_router_type_error_is_not_relabelled_or_retried():
    inner = LegacyRouter()

    class ForwardingRouter:
        def __init__(self):
            self.calls = 0

        def route(self, state, questions, **kwargs):
            self.calls += 1
            return inner.route(state, questions, **kwargs)

    router = ForwardingRouter()
    with pytest.raises(TypeError, match="unexpected keyword argument 'explain'"):
        laya_route(STATE, QUESTIONS, router=router, explain=True)
    assert router.calls == 1
    assert inner.calls == []


def test_prediction_routing_is_unchanged_after_an_explained_route():
    class PredictRouter:
        def route(self, state, questions, **kwargs):
            return dict(ROUTING, detection={"diagnostic": copy.deepcopy(DIAGNOSTIC)})

        def predict(self, state, questions, **kwargs):
            return {"answers": {"category": {"type": "choice", "choice": "A"}},
                    "routing": dict(ROUTING)}

    router = PredictRouter()
    before = json.dumps(laya_predict(STATE, QUESTIONS, router=router)["routing"])
    assert laya_route(STATE, QUESTIONS, router=router, explain=True)["diagnostic"] == DIAGNOSTIC
    after = json.dumps(laya_predict(STATE, QUESTIONS, router=router)["routing"])
    assert before == after == json.dumps(ROUTING)
    # Tournament is a real caller of the shared projection; two options need no narrowing.
    tournament = laya_tournament(STATE, QUESTIONS, router=router)
    assert json.dumps(tournament["routing"]) == json.dumps(ROUTING)


def test_mcp_route_schema_exposes_optional_boolean_explain():
    tool = next(t for t in asyncio.run(server_mod.server.list_tools()) if t.name == "laya_route")
    schema = tool.input_schema
    assert schema["properties"]["explain"]["type"] == "boolean"
    assert schema["properties"]["explain"]["default"] is False
    assert schema["required"] == ["state", "questions"]


@pytest.mark.parametrize("flag", [{}, {"explain": False}, {"explain": True}])
def test_real_mcp_route_calls_honor_boolean(flag, monkeypatch):
    router = RecordingRouter()
    monkeypatch.setattr(server_mod, "_ROUTER", router)
    out = _call_tool({"state": STATE, "questions": QUESTIONS, **flag})
    if flag.get("explain", False):
        assert out["diagnostic"] == DIAGNOSTIC
        assert router.calls[0][2] == {"explain": True}
    else:
        assert set(out) == {"model", "repo", "reason"}
        assert router.calls[0][2] == {}
    assert len(router.calls) == 1


@pytest.mark.parametrize("bad", INVALID_FLAGS)
def test_real_mcp_parser_rejects_non_boolean_without_dispatch(bad, monkeypatch):
    router = RecordingRouter()
    monkeypatch.setattr(server_mod, "_ROUTER", router)
    with pytest.raises(McpToolError, match="explain"):
        _call_tool({"state": STATE, "questions": QUESTIONS, "explain": bad})
    assert router.calls == []
