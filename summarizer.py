"""
summarizer.py
-------------
Converts a news article into a tweet-length string (≤ 240 characters).

AI backend: Google Gemini (gemini-1.5-flash)
  - Free tier: 15 requests/minute, 1 500 requests/day — plenty for a
    news bot posting 3 tweets per 30-minute run.
  - Get a free key at https://aistudio.google.com/app/apikey
  - Set the env var GEMINI_API_KEY to enable AI summarization.
  - If the key is absent the bot falls back to a fast local heuristic
    (no API needed at all).
"""

import html
import logging
import os
import re
import time

import requests

logger = logging.getLogger(__name__)

# ── Gemini API ────────────────────────────────────────────────────────────────

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL    = "gemini-1.5-flash"        # free-tier model
GEMINI_API_URL  = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
GEMINI_TIMEOUT    = 15   # seconds per attempt
GEMINI_MAX_TRIES  = 2    # attempts before falling back to local summarizer
GEMINI_BACKOFF    = 2.0  # seconds between retries

# ── tweet sizing ──────────────────────────────────────────────────────────────

TWEET_HARD_LIMIT = 280   # Twitter's absolute maximum
TWEET_SOFT_LIMIT = 240   # our target for readability


def _twitter_length(text: str) -> int:
    """
    Approximate Twitter's weighted character count.
    Characters above U+FFFF (some emoji, CJK) count as 2 units each.
    """
    return sum(2 if ord(ch) > 0xFFFF else 1 for ch in text)


# ── text-cleaning helpers ─────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Strip HTML tags, decode HTML entities, and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _truncate(text: str, max_len: int) -> str:
    """Hard-truncate so _twitter_length(result) ≤ max_len, appending '…'."""
    if _twitter_length(text) <= max_len:
        return text
    result = text
    while _twitter_length(result + "…") > max_len and result:
        result = result[:-1].rstrip()
    return result + "…"


# ── summarization strategies ──────────────────────────────────────────────────

def _local_summarize(title: str, summary: str, budget: int) -> str:
    """
    Fast heuristic – no API call needed.
    Tries: first sentence of summary → full summary → title → truncated title.
    """
    clean = _clean(summary)
    first = re.split(r"(?<=[.!?])\s", clean)[0] if clean else ""

    for candidate in (first, clean, title):
        if candidate and _twitter_length(candidate) <= budget:
            return candidate

    return _truncate(_clean(title) or title, budget)


def _ai_summarize(title: str, summary: str, budget: int) -> str:
    """
    Call Gemini to produce a concise summary within *budget* characters.
    Retries on transient failures; falls back to local summarizer otherwise.
    """
    if not GEMINI_API_KEY or len(GEMINI_API_KEY) < 10:
        logger.debug("GEMINI_API_KEY not set. Using local summarizer.")
        return _local_summarize(title, summary, budget)

    prompt = (
        f"Summarize the following news article in {budget} characters or fewer. "
        "Be concise and factual. Output only the summary — no hashtags, no emoji, "
        "no quotation marks, no preamble.\n\n"
        f"Title: {title}\n"
        f"Summary: {_clean(summary)}"
    )

    for attempt in range(1, GEMINI_MAX_TRIES + 1):
        try:
            response = requests.post(
                GEMINI_API_URL,
                params={"key": GEMINI_API_KEY},
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": 120, "temperature": 0.3},
                },
                timeout=GEMINI_TIMEOUT,
            )

            if response.status_code == 400:
                logger.error("Gemini bad request (400): %s", response.text[:200])
                return _local_summarize(title, summary, budget)

            if response.status_code == 403:
                logger.error(
                    "Gemini API key rejected (403). "
                    "Check your GEMINI_API_KEY at https://aistudio.google.com/app/apikey"
                )
                return _local_summarize(title, summary, budget)

            if response.status_code == 429:
                retry_after = float(
                    response.headers.get("retry-after", GEMINI_BACKOFF * attempt)
                )
                logger.warning(
                    "Gemini rate limit (429). Waiting %.1f s (attempt %d/%d).",
                    retry_after, attempt, GEMINI_MAX_TRIES,
                )
                time.sleep(retry_after)
                continue

            response.raise_for_status()

            data = response.json()
            text = (
                data["candidates"][0]["content"]["parts"][0]["text"]
                .strip()
                .strip('"')
            )
            return _truncate(text, budget)

        except requests.Timeout:
            logger.warning("Gemini timeout on attempt %d/%d.", attempt, GEMINI_MAX_TRIES)
            if attempt < GEMINI_MAX_TRIES:
                time.sleep(GEMINI_BACKOFF)

        except (KeyError, IndexError) as exc:
            logger.warning("Unexpected Gemini response structure: %s", exc)
            return _local_summarize(title, summary, budget)

        except Exception as exc:
            logger.warning(
                "Gemini call failed on attempt %d/%d: %s",
                attempt, GEMINI_MAX_TRIES, exc,
            )
            if attempt < GEMINI_MAX_TRIES:
                time.sleep(GEMINI_BACKOFF)

    logger.info("All Gemini attempts exhausted. Using local summarizer.")
    return _local_summarize(title, summary, budget)


# ── public API ────────────────────────────────────────────────────────────────

_PREFIX      = "🚨 Breaking: "
_INFIX       = "\n\n"
_SRC_LABEL   = "Source: "
_WRAPPER_OVERHEAD = _twitter_length(_PREFIX + _INFIX + _SRC_LABEL)


def build_tweet(article: dict) -> str:
    """
    Return a formatted tweet string for *article*.

    Format:
        🚨 Breaking: <summary>

        Source: <outlet>
    """
    source      = article.get("source", "News")
    title       = article.get("title",   "")
    summary     = article.get("summary", "")
    source_line = _SRC_LABEL + source

    budget = TWEET_SOFT_LIMIT - _WRAPPER_OVERHEAD - _twitter_length(source_line)

    body_local = _local_summarize(title, summary, budget)
    body = body_local if _twitter_length(body_local) <= budget else _ai_summarize(
        title, summary, budget
    )

    tweet = f"{_PREFIX}{body}{_INFIX}{source_line}"
    return _truncate(tweet, TWEET_HARD_LIMIT)
