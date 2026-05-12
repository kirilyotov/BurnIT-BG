"""
OPDS source implementation for book catalogs.
"""
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin
import feedparser
import requests
import time
from .base import BookSource


class OPDSSource(BookSource):
    """
    OPDS/Atom feed source for book catalogs.
    """
    # Lets at least be nice to servers and identify ourselves properly
    # https://www.rostrum.blog/posts/2019-03-04-polite-webscrape/
    USER_AGENT = "BookIngestBot/1.0 (+https://github.com/kirilyotov/BurnIT-BG)"

    def fetch_entries(
        self,
        feed_url: str,
        limit: int = 20,
        delay: float = 1.0,
        retries: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Fetch and parse entries from an OPDS/Atom feed, following pagination if needed.

        Retries on 429/5xx with exponential backoff so chitanka's Cloudflare
        rate limiter doesn't abort a multi-category run.
        """
        entries = []
        next_url = feed_url
        headers = {
            "User-Agent": self.USER_AGENT,
            "Accept": "application/atom+xml,application/xml,text/xml",
        }
        while next_url and len(entries) < limit:
            resp = None
            for attempt in range(retries):
                resp = requests.get(next_url, headers=headers, timeout=30)
                if resp.status_code < 400:
                    break
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after and retry_after.isdigit() else delay * (2 ** attempt) + 1.0
                    time.sleep(min(wait, 30.0))
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            for entry in feed.entries:
                entries.append(entry)
                if len(entries) >= limit:
                    break
            # Find next link. OPDS feeds usually advertise it as a relative
            # path (e.g. '/books/category/foo.opds/2'), so resolve against
            # the URL we just fetched before reusing it.
            next_link = None
            for link in feed.feed.get("links", []):
                if link.get("rel") == "next":
                    next_link = link.get("href")
                    break
            next_url = urljoin(next_url, next_link) if next_link else None
            time.sleep(delay)
        return entries[:limit]

    def get_source_name(self) -> str:
        """
        Return the canonical source name.
        """
        return "opds"
