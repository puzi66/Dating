"""Claude behind a small interface, so tests and offline runs use a fake.

Real client: official `anthropic` SDK (AsyncAnthropic, beta messages endpoint so the
server-side refusal fallback can be enabled). Model: CLAUDE_MODEL (default claude-sonnet-5-5).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Protocol

from .prompts import (ADI_SYSTEM_PROMPT, END_MARKER, EXTRACTION_INSTRUCTIONS, EXTRACTION_SYSTEM,
                      LIFESTYLE_FIELDS, VALUE_KEYS, WHY_SCHEMA, WHY_SYSTEM, WRAP_UP_INSTRUCTION,
                      extraction_schema, why_user_prompt)

log = logging.getLogger("app.llm")

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """Anything that went wrong talking to Claude, with a message safe to show the team."""


class LLMClient(Protocol):
    name: str

    def stream_interview(self, history: list[dict], wrap_up: bool) -> AsyncIterator[str]:
        """Yield text deltas of עדי's next message. `history` is alternating user/assistant
        {"role", "content"} dicts ending with a user turn."""
        ...

    async def extract_profile(self, transcript: str, facts: str) -> dict:
        """Return the raw profile JSON (validated later by profile.parse_profile)."""
        ...

    async def why_text(self, brief: str) -> str:
        ...


# ============================================================================ Anthropic

class AnthropicLLM:
    name = "anthropic"

    def __init__(self, api_key: str, model: str, use_fallbacks: bool = True) -> None:
        import anthropic  # imported lazily so the fake path never needs a key

        self._anthropic = anthropic
        self.client = anthropic.AsyncAnthropic(api_key=api_key, timeout=120.0, max_retries=2)
        self.model = model
        self.use_fallbacks = use_fallbacks

    def _fallback_kwargs(self) -> dict:
        # Server-side refusal fallback ("default" routing). Claude API only.
        if not self.use_fallbacks:
            return {}
        return {"betas": [FALLBACK_BETA], "fallbacks": "default"}

    async def stream_interview(self, history: list[dict], wrap_up: bool) -> AsyncIterator[str]:
        messages = list(history)
        if wrap_up:
            # Mid-conversation system message: the operator channel, appended after the last user turn.
            messages.append({"role": "system", "content": WRAP_UP_INSTRUCTION})
        try:
            async with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=2048,
                system=ADI_SYSTEM_PROMPT,
                messages=messages,
                # Conversational turn: no tools, so the lowest thinking setting keeps latency down.
                thinking={"type": "between_tools"},
                output_config={"effort": "low"},
                cache_control={"type": "ephemeral"},
                **self._fallback_kwargs(),
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
        except self._anthropic.APIConnectionError as e:
            raise LLMError("אין חיבור ל-Claude כרגע") from e
        except self._anthropic.RateLimitError as e:
            raise LLMError("יותר מדי בקשות ל-Claude, נסו שוב בעוד דקה") from e
        except self._anthropic.APIStatusError as e:
            log.error("Claude API error %s: %s", e.status_code, e.message)
            raise LLMError(f"שגיאת Claude ({e.status_code})") from e
        if final.stop_reason == "refusal":
            log.warning("interview turn refused: %s", getattr(final, "stop_details", None))
            yield "סליחה, בואו ננסה לנסח את זה אחרת. ספר/י לי עוד קצת על עצמך?"

    async def _structured(self, system: str, user: str, schema: dict, effort: str,
                          max_tokens: int) -> dict:
        try:
            async with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                thinking={"type": "adaptive"},
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
                **self._fallback_kwargs(),
            ) as stream:
                msg = await stream.get_final_message()
        except self._anthropic.APIConnectionError as e:
            raise LLMError("אין חיבור ל-Claude כרגע") from e
        except self._anthropic.RateLimitError as e:
            raise LLMError("יותר מדי בקשות ל-Claude, נסו שוב בעוד דקה") from e
        except self._anthropic.APIStatusError as e:
            log.error("Claude API error %s: %s", e.status_code, e.message)
            raise LLMError(f"שגיאת Claude ({e.status_code})") from e
        if msg.stop_reason == "refusal":
            raise LLMError("Claude סירב לבקשה")
        if msg.stop_reason == "max_tokens":
            raise LLMError("התשובה של Claude נקטעה")
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError("Claude החזיר JSON לא תקין") from e

    async def extract_profile(self, transcript: str, facts: str) -> dict:
        user = (f"<basic_facts>\n{facts}\n</basic_facts>\n\n"
                f"<interview_transcript>\n{transcript}\n</interview_transcript>\n\n{EXTRACTION_INSTRUCTIONS}")
        return await self._structured(EXTRACTION_SYSTEM, user, extraction_schema(), "medium", 16000)

    async def why_text(self, brief: str) -> str:
        data = await self._structured(WHY_SYSTEM, why_user_prompt(brief), WHY_SCHEMA, "low", 4000)
        return str(data.get("text", "")).strip()


# ============================================================================ Fake

FAKE_QUESTIONS = [
    "היי, אני עדי, השדכנית של שלוש. נדבר בערך רבע שעה כדי שאכיר אותך באמת, ואפשר לענות בחופשיות ובאורך שנוח לך. "
    "בוא/י נתחיל ממשהו קליל: איך נראה סוף שבוע טוב בשבילך?",
    "נשמע כיף. ומה ממלא אותך באמצע השבוע, בעבודה או מחוצה לה?",
    "אם היית צריך/ה לבחור דבר אחד שאת/ה הכי גאה בו בעצמך, מה זה היה?",
    "ומה הכי מרגיז אותך אצל אנשים אחרים?",
    "ספר/י לי על קשר קודם: מה עבד בו, ומה למדת על עצמך?",
    "איך נראה בשבילך הקשר שאת/ה מחפש/ת עכשיו?",
]
FAKE_CLOSING = ("תודה רבה, היה ממש כיף להכיר אותך. אני מסכמת עכשיו את מה שסיפרת, "
                "והצוות יעבור על הפרופיל לפני שההצעות הראשונות יוצאות. " + END_MARKER)


class FakeLLM:
    """Deterministic stand-in for Claude: scripted עדי, hash-based profile, template why-text."""

    name = "fake"

    def __init__(self, delay: float = 0.0, max_questions: int | None = None) -> None:
        self.delay = delay
        self.max_questions = max_questions or len(FAKE_QUESTIONS)
        self.calls: list[tuple[str, object]] = []
        self.extract_response: dict | str | None = None   # tests can override

    async def stream_interview(self, history: list[dict], wrap_up: bool) -> AsyncIterator[str]:
        self.calls.append(("interview", len(history)))
        n_user = sum(1 for m in history if m["role"] == "user")
        text = FAKE_CLOSING if (wrap_up or n_user > self.max_questions) else FAKE_QUESTIONS[n_user - 1]
        for i in range(0, len(text), 6):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield text[i:i + 6]

    async def extract_profile(self, transcript: str, facts: str) -> dict:
        self.calls.append(("extract", len(transcript)))
        if self.extract_response is not None:
            return self.extract_response  # type: ignore[return-value]
        h = hashlib.sha256(transcript.encode()).digest()
        values = {k: (h[i] % 11) for i, k in enumerate(VALUE_KEYS)}
        return {
            "summary": "אדם חם וסקרן, אוהב טבע ואנשים, מחפש/ת קשר יציב עם הרבה צחוק. (פרופיל לדוגמה, ללא Claude)",
            "values": values,
            "lifestyle": {k: "לא עלה בשיחה" for k in LIFESTYLE_FIELDS},
            "looking_for": ["מישהו/י עם הומור", "שיחות עמוקות עד מאוחר", "שותפות אמיתית ביומיום"],
            "red_flags": [],
        }

    async def why_text(self, brief: str) -> str:
        self.calls.append(("why", len(brief)))
        return ("לשניכם יש אותה גישה לחיים: סקרנות, הומור ורצון בקשר אמיתי. "
                "נראה לי ששיחה ראשונה ביניכם תזרום בקלות.")


def make_llm(settings) -> LLMClient:  # noqa: ANN001
    if settings.llm_enabled:
        return AnthropicLLM(settings.anthropic_api_key, settings.claude_model,
                            use_fallbacks=settings.claude_fallbacks)
    log.warning("ANTHROPIC_API_KEY missing or FAKE_LLM=1: using the fake Claude client")
    return FakeLLM(delay=0.03)


class MarkerFilter:
    """Strips END_MARKER from a token stream even when it is split across chunks."""

    def __init__(self, marker: str = END_MARKER) -> None:
        self.marker = marker
        self.buf = ""
        self.found = False

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        if self.marker in self.buf:
            self.found = True
            out = self.buf.replace(self.marker, "")
            self.buf = ""
            return out
        keep = 0
        for k in range(min(len(self.marker) - 1, len(self.buf)), 0, -1):
            if self.marker.startswith(self.buf[-k:]):
                keep = k
                break
        out, self.buf = (self.buf[:-keep], self.buf[-keep:]) if keep else (self.buf, "")
        return out

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out
