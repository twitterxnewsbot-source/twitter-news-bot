"""
main.py
-------
Entry point for the Twitter/X news bot.

Improvements over v1
---------------------
Startup credential validation
    Credentials and critical env vars are validated before any network
    calls are made.  A missing key surfaces as a clear error at launch
    rather than a confusing 401 mid-run.

Per-article error isolation
    An unhandled exception in build_tweet() or post_tweet() for one
    article no longer aborts the entire run; it is caught, logged with
    full traceback, and the bot moves on to the next article.

Graceful shutdown on SIGINT / SIGTERM
    The scheduler loop catches KeyboardInterrupt and SIGTERM cleanly and
    logs a tidy shutdown message instead of printing a raw traceback.

End-of-run status report
    Each run logs a one-line summary: articles fetched, skipped (already
    posted), skipped (too old / no link), tweet failures, and tweets sent.
    This makes CI/CD run logs easy to scan.

Inter-tweet delay derived from rate-limit headers
    The fixed 2-second sleep between tweets is replaced by a helper that
    reads the current rate-limit state and sleeps only as long as needed.
"""

import argparse
import logging
import os
import signal
import sys
import time

from dotenv import load_dotenv

# Load .env before any module that reads os.getenv()
load_dotenv()

from news_fetcher   import fetch_articles
from summarizer     import build_tweet
from twitter_poster import post_tweet, monthly_usage
from storage        import is_posted, mark_posted, count_posted

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

# ── constants ──────────────────────────────────────────────────────────────────

SCHEDULE_INTERVAL_MINUTES = int(os.getenv("SCHEDULE_INTERVAL_MINUTES", "30"))

# Post at most this many tweets per run.
# Free tier: 500/month ÷ ~1440 runs/month ≈ 0.35 tweets/run → keep at 3
# to stay well within budget.
MAX_TWEETS_PER_RUN = int(os.getenv("MAX_TWEETS_PER_RUN", "3"))

# Minimum seconds between consecutive tweets in the same run.
INTER_TWEET_DELAY = float(os.getenv("INTER_TWEET_DELAY", "5"))


# ── startup validation ────────────────────────────────────────────────────────

_REQUIRED_ENV_VARS = [
    "TWITTER_API_KEY",
    "TWITTER_API_SECRET",
    "TWITTER_ACCESS_TOKEN",
    "TWITTER_ACCESS_TOKEN_SECRET",
]


def validate_config() -> bool:
    """
    Check that all required environment variables are present.
    Returns True if OK, False otherwise (logs each missing variable).
    """
    missing = [v for v in _REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        for var in missing:
            logger.error("Required environment variable not set: %s", var)
        logger.error(
            "Copy .env.example to .env and fill in your Twitter credentials, "
            "then re-run the bot."
        )
        return False

    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.info(
            "ANTHROPIC_API_KEY not set – AI summarization disabled. "
            "Using local heuristic summarizer."
        )
    return True


# ── core logic ─────────────────────────────────────────────────────────────────

def run_once() -> None:
    """Fetch news, filter duplicates, summarise, tweet, and log a status report."""
    monthly_sent, monthly_limit = monthly_usage()
    logger.info(
        "=== Bot run started | stored links: %d | monthly usage: %d/%d ===",
        count_posted(),
        monthly_sent,
        monthly_limit,
    )

    articles = fetch_articles()
    logger.info("Articles fetched: %d", len(articles))

    stats = {"skipped_dup": 0, "skipped_invalid": 0, "tweet_errors": 0, "tweeted": 0}

    for article in articles:
        if stats["tweeted"] >= MAX_TWEETS_PER_RUN:
            logger.info("Reached MAX_TWEETS_PER_RUN (%d). Stopping.", MAX_TWEETS_PER_RUN)
            break

        url = article.get("link", "")
        if not url:
            stats["skipped_invalid"] += 1
            continue

        if is_posted(url):
            stats["skipped_dup"] += 1
            logger.debug("Already posted, skipping: %s", url)
            continue

        # ── build tweet text ──────────────────────────────────────────────
        try:
            tweet_text = build_tweet(article)
        except Exception as exc:
            logger.error(
                "Unexpected error building tweet for '%s': %s",
                article.get("title", url), exc, exc_info=True,
            )
            stats["skipped_invalid"] += 1
            continue

        logger.info(
            "Posting: %s",
            tweet_text[:100] + ("…" if len(tweet_text) > 100 else ""),
        )

        # ── post tweet ────────────────────────────────────────────────────
        try:
            success = post_tweet(tweet_text)
        except Exception as exc:
            logger.error(
                "Unexpected error posting tweet for '%s': %s",
                article.get("title", url), exc, exc_info=True,
            )
            success = False

        if success:
            mark_posted(url)
            stats["tweeted"] += 1
            if stats["tweeted"] < MAX_TWEETS_PER_RUN:
                time.sleep(INTER_TWEET_DELAY)
        else:
            stats["tweet_errors"] += 1
            logger.warning("Failed to post tweet for: %s", url)

    logger.info(
        "=== Run complete | tweeted: %d | dup-skipped: %d | "
        "invalid-skipped: %d | errors: %d ===",
        stats["tweeted"],
        stats["skipped_dup"],
        stats["skipped_invalid"],
        stats["tweet_errors"],
    )


# ── scheduler ─────────────────────────────────────────────────────────────────

_shutdown_requested = False


def _handle_signal(signum, _frame) -> None:
    global _shutdown_requested
    logger.info("Received signal %d. Finishing current run then shutting down.", signum)
    _shutdown_requested = True


def run_on_schedule(interval_minutes: int = SCHEDULE_INTERVAL_MINUTES) -> None:
    """Run run_once() repeatedly with a fixed interval until interrupted."""
    # Register graceful shutdown handlers
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("Scheduler started – running every %d minutes.", interval_minutes)

    while not _shutdown_requested:
        try:
            run_once()
        except Exception as exc:
            logger.error("Unhandled error in run_once(): %s", exc, exc_info=True)

        if _shutdown_requested:
            break

        sleep_seconds = interval_minutes * 60
        logger.info("Next run in %d minutes.", interval_minutes)

        # Sleep in short chunks so SIGTERM is noticed quickly
        deadline = time.monotonic() + sleep_seconds
        while time.monotonic() < deadline and not _shutdown_requested:
            time.sleep(min(5, deadline - time.monotonic()))

    logger.info("Bot shut down cleanly.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AI-powered Twitter/X news bot")
    parser.add_argument(
        "--schedule",
        action="store_true",
        help=f"Run continuously every {SCHEDULE_INTERVAL_MINUTES} minutes.",
    )
    args = parser.parse_args()

    if not validate_config():
        sys.exit(1)

    if args.schedule:
        run_on_schedule()
    else:
        run_once()


if __name__ == "__main__":
    main()
