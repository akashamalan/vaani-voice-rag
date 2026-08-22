"""
Sarvam speech-to-text.

Audio goes browser -> our backend -> Sarvam. Never browser -> Sarvam: the
live link is public, and a key in browser JS is a key that is gone.

The httpx client is created once at startup and passed in. Building a TLS
connection per request costs 100ms+ and would land straight on the latency
panel.
"""
import base64

import httpx

from app.config import (HTTP_TIMEOUT, RETRY_ATTEMPTS, SARVAM_LANG,
                        SARVAM_MODEL, SARVAM_URL, sarvam_key)
from app.harness.retry import with_retry


class SarvamSTT:
    def __init__(self, client: httpx.AsyncClient, lang: str = "hi"):
        self.client = client
        self.lang = lang
        self._key = sarvam_key()          # fail at startup, not first request

    @staticmethod
    def make_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    async def transcribe(self, audio: bytes | str, mime: str = "audio/wav") -> str:
        """
        audio: raw bytes, or a base64 string as it arrives over the socket.
        Returns the transcript, or "" if Sarvam heard nothing.
        """
        if isinstance(audio, str):
            audio = base64.b64decode(audio)

        async def call():
            r = await self.client.post(
                SARVAM_URL,
                headers={"api-subscription-key": self._key},
                files={"file": ("audio.wav", audio, mime)},
                data={"model": SARVAM_MODEL,
                      "language_code": SARVAM_LANG.get(self.lang, "hi-IN")},
            )
            r.raise_for_status()
            return r.json()

        payload = await with_retry(call, attempts=RETRY_ATTEMPTS, what="sarvam stt")
        return (payload.get("transcript") or "").strip()
