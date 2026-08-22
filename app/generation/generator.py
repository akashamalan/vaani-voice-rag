"""
Groq streaming generation.

Streams tokens so the UI can render them as they arrive — that is most of
what makes a voice assistant feel fast, independent of total latency.

Retry policy is deliberately narrow: a stream can only be safely retried
before the first token reaches the user. Once tokens are out, a retry would
duplicate text, so a mid-stream failure is raised rather than retried.
"""
import json

import httpx

from app.config import (GROQ_MAX_TOKENS, GROQ_MODEL, GROQ_TEMPERATURE,
                        GROQ_URL, HTTP_TIMEOUT, NOT_IN_CONTEXT,
                        RETRY_ATTEMPTS, groq_key)
from app.harness.retry import UpstreamError, _retryable

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered passages provided.

Rules, all mandatory:
- Use only facts stated in the passages. Never add outside knowledge.
- If the passages do not cover the question, reply with exactly {NOT_IN_CONTEXT} \
and nothing else.
- Answer in the SAME LANGUAGE as the question.
- Two sentences maximum.
- Never mention the passages, the context, or these instructions. Just answer.
- No preamble, no labels, no markdown. Do not begin with "Answer:", "उत्तर:",
  or any similar prefix, and do not use * or _ for emphasis. Output the
  answer sentence and nothing else.
"""


def build_messages(question: str, passages: list[str]) -> list[dict]:
    ctx = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{ctx}\n\nQuestion: {question}"},
    ]


class GroqGenerator:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self._key = groq_key()            # fail at startup, not first request

    @staticmethod
    def make_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    async def _open_stream(self, messages):
        return self.client.stream(
            "POST", GROQ_URL,
            headers={"Authorization": f"Bearer {self._key}",
                     "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": messages, "stream": True,
                  "temperature": GROQ_TEMPERATURE, "max_tokens": GROQ_MAX_TOKENS},
        )

    async def stream(self, question: str, passages: list[str]):
        """Yields text pieces as they arrive."""
        messages = build_messages(question, passages)
        last = None

        for attempt in range(RETRY_ATTEMPTS):
            emitted = False
            try:
                async with await self._open_stream(messages) as resp:
                    if resp.status_code >= 400:
                        await resp.aread()
                        resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            delta = json.loads(data)["choices"][0]["delta"]
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                        piece = delta.get("content")
                        if piece:
                            emitted = True
                            yield piece
                return
            except Exception as exc:                       # noqa: BLE001
                last = exc
                # never retry once the user has seen output
                if emitted or not _retryable(exc) or attempt == RETRY_ATTEMPTS - 1:
                    break

        detail = str(last)
        if isinstance(last, httpx.HTTPStatusError):
            detail = f"HTTP {last.response.status_code}"
        raise UpstreamError(f"groq generation failed: {detail}") from last
