"""The real AnthropicLLM against a mocked HTTP transport (no API key, no network).

This checks that our calls go through the official SDK with the intended request shape
(model, streaming, structured output, fallbacks beta) and that responses are parsed.
It cannot check that the live API accepts every parameter; see README caveats.
"""

from __future__ import annotations

import asyncio
import json

import httpx2
import pytest

from app.llm import FALLBACK_BETA, AnthropicLLM, LLMError, MarkerFilter
from app.prompts import END_MARKER, VALUE_KEYS


def _sse_message(text_chunks: list[str], stop_reason: str = "end_turn") -> bytes:
    events = [("message_start", {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}})]
    for t in text_chunks:
        events.append(("content_block_delta", {"type": "content_block_delta", "index": 0,
                                               "delta": {"type": "text_delta", "text": t}}))
    events += [("content_block_stop", {"type": "content_block_stop", "index": 0}),
               ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason,
                                                                       "stop_sequence": None},
                                  "usage": {"output_tokens": 5}}),
               ("message_stop", {"type": "message_stop"})]
    return "".join(f"event: {e}\ndata: {json.dumps(d, ensure_ascii=False)}\n\n" for e, d in events).encode()


class Recorder:
    def __init__(self, chunks: list[str], stop_reason: str = "end_turn", status: int = 200) -> None:
        self.chunks, self.stop_reason, self.status = chunks, stop_reason, status
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx2.Response(self.status, json={"type": "error", "error": {"type": "invalid_request_error",
                                                                                 "message": "bad"}})
        return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                               content=_sse_message(self.chunks, self.stop_reason))

    def body(self, i: int = -1) -> dict:
        return json.loads(self.requests[i].content)


def _llm(rec: Recorder) -> AnthropicLLM:
    return AnthropicLLM("sk-test", "claude-sonnet-5-5",
                        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(rec)))


async def _collect(llm: AnthropicLLM, history, wrap_up=False) -> str:
    return "".join([t async for t in llm.stream_interview(history, wrap_up)])


def test_interview_stream_request_shape_and_text():
    rec = Recorder(["היי, אני ", "עדי. ", "מה נשמע?"])
    out = asyncio.run(_collect(_llm(rec), [{"role": "user", "content": "שלום"}]))
    assert out == "היי, אני עדי. מה נשמע?"
    req, body = rec.requests[0], rec.body()
    assert req.url.path == "/v1/messages" and req.headers["x-api-key"] == "sk-test"
    assert FALLBACK_BETA in req.headers.get("anthropic-beta", "")
    assert body["model"] == "claude-sonnet-5-5" and body["stream"] is True
    assert body["fallbacks"] == "default" and body["thinking"] == {"type": "between_tools"}
    assert body["output_config"] == {"effort": "low"} and "עדי" in body["system"]
    assert body["messages"] == [{"role": "user", "content": "שלום"}]


def test_interview_wrap_up_appends_system_message_and_marker_is_filterable():
    rec = Recorder(["תודה רבה! ", "[[EN", "D]]"])
    llm = _llm(rec)
    flt = MarkerFilter()

    async def run() -> str:
        return "".join([flt.feed(t) async for t in llm.stream_interview([{"role": "user", "content": "x"}], True)])
    out = asyncio.run(run()) + flt.flush()
    assert out == "תודה רבה! " and flt.found
    msgs = rec.body()["messages"]
    assert msgs[-1]["role"] == "system" and END_MARKER in msgs[-1]["content"]


def test_structured_extraction_request_and_parse():
    payload = {"summary": "s", "values": {k: 5 for k in VALUE_KEYS}, "lifestyle": {}, "looking_for": ["a"],
               "red_flags": []}
    rec = Recorder([json.dumps(payload, ensure_ascii=False)])
    data = asyncio.run(_llm(rec).extract_profile("תמליל", "עובדות"))
    assert data == payload
    body = rec.body()
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert body["thinking"] == {"type": "adaptive"} and "תמליל" in body["messages"][0]["content"]


def test_why_text_and_errors():
    rec = Recorder([json.dumps({"text": " לשניכם יש הומור "})])
    assert asyncio.run(_llm(rec).why_text("brief")) == "לשניכם יש הומור"
    with pytest.raises(LLMError):
        asyncio.run(_llm(Recorder(["{}"], stop_reason="refusal")).why_text("brief"))
    with pytest.raises(LLMError):
        asyncio.run(_llm(Recorder(["not json"])).extract_profile("t", "f"))
    with pytest.raises(LLMError):
        asyncio.run(_llm(Recorder([], status=400)).why_text("brief"))
