"""
Exponential backoff with jitter.

The important part is what we do NOT retry. A 400 or a 401 will fail again
in exactly the same way, and each retry costs the user another round trip.
So: retry timeouts, connection errors, 5xx, and 429. Give up immediately on
every other 4xx.
"""
import asyncio
import random

import httpx


class UpstreamError(RuntimeError):
    """Any upstream failure that survived the retry policy."""


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 429:
            return True
        if 400 <= code < 500:
            return False        # malformed / unauthorized — retrying is waste
        return code >= 500
    return False


async def with_retry(fn, *, attempts: int = 3, base: float = 0.25,
                     cap: float = 4.0, what: str = "upstream"):
    """
    Call an async fn, retrying on transient failures only.
    `fn` takes no arguments — close over what it needs.
    """
    last = None
    tried = 0
    for i in range(attempts):
        tried = i + 1
        try:
            return await fn()
        except Exception as exc:                     # noqa: BLE001
            last = exc
            if not _retryable(exc) or i == attempts - 1:
                break
            delay = min(cap, base * (2 ** i))
            delay += random.uniform(0, delay * 0.25)   # jitter
            await asyncio.sleep(delay)

    detail = str(last)
    if isinstance(last, httpx.HTTPStatusError):
        detail = f"HTTP {last.response.status_code}: {last.response.text[:200]}"
    raise UpstreamError(f"{what} failed after {tried} attempt(s): {detail}") from last
