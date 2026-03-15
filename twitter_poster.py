"""
twitter_poster.py
-----------------
Authenticates with Twitter/X via Tweepy and posts tweets safely.

Improvements over v1
---------------------
Rate-limit awareness
    * Reads the x-rate-limit-remaining / x-rate-limit-reset headers from
      every response and records them in the module-level RateLimitState.
    * If the remaining budget drops to 0 the call blocks until the reset
      window passes before retrying, instead of getting a hard 429.

Monthly budget guard
    * Twitter's free tier allows 500 tweets/month.  A lightweight monthly
      counter is kept in a sidecar JSON file.  When the counter reaches
      MONTHLY_TWEET_LIMIT the poster refuses to post and logs a clear
      warning, preventing accidental over-spend.

Retry with exponential back-off
    * Transient errors (network timeouts, HTTP 5xx, recoverable Tweepy
      errors) are retried up to MAX_RETRIES times with jittered back-off.
    * Permanent errors (bad credentials, duplicate content, suspended
      account) are never retried – the failure is returned immediately.

Single cached client
    * The tweepy.Client is built once and reused, avoiding repeated
      credential validation overhead.
"""

import json
import logging
import os
import time
import random
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import tweepy

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

# Twitter free-tier monthly write quota.  Raise for Basic ($100/mo = 3,000).
MONTHLY_TWEET_LIMIT = int(os.getenv("MONTHLY_TWEET_LIMIT", "450"))  # 10% safety margin

# Retry settings for transient failures.
MAX_RETRIES   = 3
BASE_BACKOFF  = 2.0   # seconds; doubles each attempt + jitter

# HTTP status codes that are permanent failures (no point retrying).
PERMANENT_HTTP_ERRORS = {400, 401, 403}

# Tweepy error codes that indicate permanent failures.
# 187 = duplicate tweet, 226 = automated-behaviour detection,
# 261 = app suspended, 326 = account locked.
PERMANENT_TWEEPY_CODES = {187, 226, 261, 326}

# Path to the monthly-usage sidecar file.
_DEFAULT_USAGE_PATH = Path(__file__).parent / "tweet_usage.json"
USAGE_FILE = Path(os.getenv("TWEET_USAGE_FILE", str(_DEFAULT_USAGE_PATH)))


# ── rate-limit state ──────────────────────────────────────────────────────────

@dataclass
class RateLimitState:
    """Tracks the most recent rate-limit headers from Twitter."""
    remaining: int   = 999
    reset_at:  float = 0.0   # Unix timestamp when the window resets


_rl = RateLimitState()


def _update_rate_limit(response: tweepy.Response) -> None:
    """
    Parse rate-limit headers from a Tweepy Response object.
    Tweepy surfaces them via response.headers (when return_type=Response).
    """
    headers = getattr(response, "headers", None) or {}
    try:
        if "x-rate-limit-remaining" in headers:
            _rl.remaining = int(headers["x-rate-limit-remaining"])
        if "x-rate-limit-reset" in headers:
            _rl.reset_at = float(headers["x-rate-limit-reset"])
    except (ValueError, TypeError):
        pass   # malformed headers – ignore


def _wait_if_rate_limited() -> None:
    """Block until the rate-limit window resets if we're out of quota."""
    if _rl.remaining <= 1:
        now   = time.time()
        sleep = max(0.0, _rl.reset_at - now) + 1.0   # +1 s safety margin
        if sleep > 1:
            logger.warning(
                "Rate limit nearly exhausted. Waiting %.0f s for window reset.",
                sleep,
            )
            time.sleep(sleep)


# ── monthly budget ────────────────────────────────────────────────────────────

def _load_usage() -> dict:
    if not USAGE_FILE.exists():
        return {}
    try:
        with USAGE_FILE.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_usage(data: dict) -> None:
    try:
        USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=USAGE_FILE.parent, prefix=".tweet_usage_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp, USAGE_FILE)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError as exc:
        logger.error("Could not save usage file: %s", exc)


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _check_and_increment_budget() -> bool:
    """
    Return True if we are within the monthly budget and increment the counter.
    Return False and log a warning if the budget is exhausted.
    """
    month = _current_month()
    usage = _load_usage()
    count = usage.get(month, 0)

    if count >= MONTHLY_TWEET_LIMIT:
        logger.error(
            "Monthly tweet budget exhausted (%d / %d for %s). "
            "Skipping post. Raise MONTHLY_TWEET_LIMIT or wait until next month.",
            count,
            MONTHLY_TWEET_LIMIT,
            month,
        )
        return False

    usage[month] = count + 1
    _save_usage(usage)
    logger.debug("Monthly usage: %d / %d (%s)", count + 1, MONTHLY_TWEET_LIMIT, month)
    return True


# ── client singleton ──────────────────────────────────────────────────────────

_client: "tweepy.Client | None" = None


def _get_client() -> tweepy.Client:
    global _client
    if _client is not None:
        return _client

    required = {
        "TWITTER_API_KEY":             os.getenv("TWITTER_API_KEY"),
        "TWITTER_API_SECRET":          os.getenv("TWITTER_API_SECRET"),
        "TWITTER_ACCESS_TOKEN":        os.getenv("TWITTER_ACCESS_TOKEN"),
        "TWITTER_ACCESS_TOKEN_SECRET": os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(
            f"Missing Twitter credentials in environment: {', '.join(missing)}"
        )

    _client = tweepy.Client(
        consumer_key        = required["TWITTER_API_KEY"],
        consumer_secret     = required["TWITTER_API_SECRET"],
        access_token        = required["TWITTER_ACCESS_TOKEN"],
        access_token_secret = required["TWITTER_ACCESS_TOKEN_SECRET"],
        return_type         = tweepy.Response,  # gives us access to headers
        wait_on_rate_limit  = False,            # we handle this ourselves
    )
    return _client


# ── public API ────────────────────────────────────────────────────────────────

def post_tweet(text: str) -> bool:
    """
    Post *text* as a new tweet.

    Returns True on success, False on any unrecoverable failure.
    Handles rate limits, retries transient errors, and enforces the
    monthly budget cap.
    """
    # 1. Monthly budget guard (cheap, no network call)
    if not _check_and_increment_budget():
        return False

    attempt = 0
    while attempt <= MAX_RETRIES:
        # 2. Honour rate-limit window before every attempt
        _wait_if_rate_limited()

        try:
            client   = _get_client()
            response = client.create_tweet(text=text)

            # 3. Update rate-limit state from response headers
            _update_rate_limit(response)

            tweet_id = response.data.get("id", "unknown")
            logger.info("Tweet posted successfully (id=%s).", tweet_id)
            return True

        # ── permanent Twitter API errors – never retry ─────────────────────
        except tweepy.Forbidden as exc:
            code = _tweepy_error_code(exc)
            logger.error(
                "Permanent Twitter error (403, code=%s): %s. Not retrying.",
                code, exc,
            )
            # Undo the budget increment we made above – tweet wasn't sent.
            _decrement_budget()
            return False

        except tweepy.Unauthorized as exc:
            logger.error("Twitter credentials rejected (401): %s", exc)
            _decrement_budget()
            return False

        except tweepy.BadRequest as exc:
            logger.error("Bad request (400): %s", exc)
            _decrement_budget()
            return False

        # ── rate limit hit despite our proactive check ─────────────────────
        except tweepy.TooManyRequests as exc:
            reset_header = getattr(exc, "response", None)
            reset_ts     = 0.0
            if reset_header is not None:
                try:
                    reset_ts = float(reset_header.headers.get("x-rate-limit-reset", 0))
                except (ValueError, TypeError, AttributeError):
                    pass

            wait = max(60.0, reset_ts - time.time()) + 1.0
            logger.warning(
                "429 Too Many Requests. Waiting %.0f s before retry %d/%d.",
                wait, attempt + 1, MAX_RETRIES,
            )
            time.sleep(wait)
            attempt += 1
            continue

        # ── transient errors – retry with exponential back-off ─────────────
        except (tweepy.TwitterServerError, tweepy.TweepyException) as exc:
            if attempt >= MAX_RETRIES:
                logger.error(
                    "Failed to post tweet after %d attempts: %s", MAX_RETRIES + 1, exc
                )
                _decrement_budget()
                return False

            backoff = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                "Transient error (%s). Retrying in %.1f s (attempt %d/%d).",
                exc, backoff, attempt + 1, MAX_RETRIES,
            )
            time.sleep(backoff)
            attempt += 1

        except RuntimeError as exc:
            # Credential configuration error – no point retrying
            logger.error("Configuration error: %s", exc)
            _decrement_budget()
            return False

        except Exception as exc:
            logger.error("Unexpected error posting tweet: %s", exc, exc_info=True)
            _decrement_budget()
            return False

    # Exhausted all retries
    _decrement_budget()
    return False


def _tweepy_error_code(exc: tweepy.TweepyException) -> "int | None":
    """Extract the Twitter API error code from a TweepyException, if present."""
    try:
        errors = exc.api_errors  # type: ignore[attr-defined]
        if errors:
            return errors[0].get("code")
    except AttributeError:
        pass
    return None


def _decrement_budget() -> None:
    """Roll back a budget increment when a tweet ultimately wasn't sent."""
    month = _current_month()
    usage = _load_usage()
    if usage.get(month, 0) > 0:
        usage[month] -= 1
        _save_usage(usage)


def monthly_usage() -> tuple[int, int]:
    """Return (tweets_sent_this_month, monthly_limit) for status reporting."""
    month = _current_month()
    count = _load_usage().get(month, 0)
    return count, MONTHLY_TWEET_LIMIT
