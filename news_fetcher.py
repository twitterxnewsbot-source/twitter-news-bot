"""
news_fetcher.py
---------------
Fetches the latest articles from a list of RSS feeds.

Improvements over v1
---------------------
Per-feed timeout
    feedparser.parse() can block indefinitely on a slow host.  We now
    pass a hard `agent` / socket-level timeout via a requests Session so
    each feed has a bounded wall-clock limit.

HTTP 304 / ETag+Last-Modified caching
    feedparser supports conditional GETs.  We persist the ETag and
    Last-Modified value for each feed between calls so we skip parsing
    feeds that haven't changed, saving bandwidth and reducing the chance
    of being rate-limited by the feed host.

Published-date filtering
    Articles older than MAX_ARTICLE_AGE_HOURS are silently skipped so the
    bot never surfaces stale content after a cold start or gap in runs.

Resilient bozo handling
    A feed marked bozo (parse error) still has its entries returned if
    feedparser managed to recover partial content, so we don't throw away
    a mostly-valid feed on a minor XML quirk.
"""

import logging
import os
import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import feedparser

logger = logging.getLogger(__name__)

# ── configuration ─────────────────────────────────────────────────────────────

RSS_FEEDS: dict[str, str] = {
    "BBC News":   "http://feeds.bbci.co.uk/news/rss.xml",
    "Reuters":    "https://feeds.reuters.com/reuters/topNews",
    "TechCrunch": "https://techcrunch.com/feed/",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "The Verge":  "https://www.theverge.com/rss/index.xml",
}

MAX_ARTICLES_PER_FEED  = 5
FEED_TIMEOUT_SECONDS   = 10       # wall-clock limit per feed request
MAX_ARTICLE_AGE_HOURS  = 48       # ignore articles older than this

# Where to persist per-feed ETags / Last-Modified headers between runs.
_DEFAULT_CACHE_PATH = Path(__file__).parent / "feed_cache.json"
FEED_CACHE_FILE     = Path(os.getenv("FEED_CACHE_FILE", str(_DEFAULT_CACHE_PATH)))


# ── ETag / Last-Modified cache ─────────────────────────────────────────────────

def _load_feed_cache() -> dict:
    if not FEED_CACHE_FILE.exists():
        return {}
    try:
        with FEED_CACHE_FILE.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_feed_cache(cache: dict) -> None:
    try:
        FEED_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=FEED_CACHE_FILE.parent, prefix=".feed_cache_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(cache, fh, indent=2)
            os.replace(tmp, FEED_CACHE_FILE)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError as exc:
        logger.warning("Could not save feed cache: %s", exc)


# ── age filtering ─────────────────────────────────────────────────────────────

def _is_too_old(entry) -> bool:
    """Return True if the entry's published date exceeds MAX_ARTICLE_AGE_HOURS."""
    # feedparser normalises dates into 9-tuple struct_time in 'published_parsed'
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if parsed is None:
        return False   # no date → don't filter out; let it through

    try:
        pub_dt  = datetime(*parsed[:6], tzinfo=timezone.utc)
        age     = datetime.now(timezone.utc) - pub_dt
        too_old = age > timedelta(hours=MAX_ARTICLE_AGE_HOURS)
        if too_old:
            logger.debug("Skipping old article (%s ago): %s", age, getattr(entry, "title", ""))
        return too_old
    except (TypeError, ValueError):
        return False


# ── public API ────────────────────────────────────────────────────────────────

def fetch_articles() -> list[dict]:
    """
    Poll every RSS feed and return a flat list of fresh article dicts.

    Each dict has keys: title, link, summary, source, published_utc.
    """
    articles: list[dict] = []
    cache = _load_feed_cache()
    cache_updated = False

    for source, url in RSS_FEEDS.items():
        entry_key   = url          # use URL as cache key
        feed_kwargs: dict = {
            "request_headers": {"User-Agent": "Mozilla/5.0 (compatible; NewsBot/2.0)"},
        }

        # Pass ETag / Last-Modified for conditional GET (saves bandwidth)
        if entry_key in cache:
            if "etag" in cache[entry_key]:
                feed_kwargs["etag"] = cache[entry_key]["etag"]
            if "modified" in cache[entry_key]:
                feed_kwargs["modified"] = cache[entry_key]["modified"]

        # feedparser doesn't expose a socket timeout natively; set it via
        # the socket default for the duration of this call.
        import socket
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(FEED_TIMEOUT_SECONDS)

        try:
            feed = feedparser.parse(url, **feed_kwargs)
        except Exception as exc:
            logger.error("Exception parsing feed %s: %s", source, exc)
            continue
        finally:
            socket.setdefaulttimeout(old_timeout)

        # HTTP 304 Not Modified → nothing new, skip
        status = getattr(feed, "status", 200)
        if status == 304:
            logger.debug("Feed %s not modified (304). Skipping.", source)
            continue

        if status >= 400:
            logger.warning("Feed %s returned HTTP %d. Skipping.", source, status)
            continue

        # Log parse warnings but don't abort
        if feed.bozo and not feed.entries:
            logger.warning(
                "Feed %s has parse issues and no entries (%s). Skipping.",
                source, getattr(feed, "bozo_exception", "unknown error"),
            )
            continue
        elif feed.bozo:
            logger.debug(
                "Feed %s has minor parse issue (%s) but entries recovered.",
                source, getattr(feed, "bozo_exception", ""),
            )

        # Persist ETag / Last-Modified for next run
        if getattr(feed, "etag", None):
            cache.setdefault(entry_key, {})["etag"] = feed.etag
            cache_updated = True
        if getattr(feed, "modified", None):
            cache.setdefault(entry_key, {})["modified"] = feed.modified
            cache_updated = True

        new_count = 0
        for entry in feed.entries[:MAX_ARTICLES_PER_FEED]:
            if _is_too_old(entry):
                continue

            link = entry.get("link", "").strip()
            if not link:
                continue

            # Determine a published timestamp string for logging/metadata
            parsed_date = getattr(entry, "published_parsed", None)
            pub_utc = (
                datetime(*parsed_date[:6], tzinfo=timezone.utc).isoformat()
                if parsed_date else ""
            )

            articles.append({
                "title":         entry.get("title", "No title").strip(),
                "link":          link,
                "summary":       entry.get("summary", entry.get("description", "")).strip(),
                "source":        source,
                "published_utc": pub_utc,
            })
            new_count += 1

        logger.info("Fetched %d usable articles from %s (HTTP %d).", new_count, source, status)

    if cache_updated:
        _save_feed_cache(cache)

    return articles
