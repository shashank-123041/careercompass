"""
events.py — Tech fests, competitions and workshops scraper
Sources:
  Tech Fests:  Unstop competitions (api.unstop.com) + Unstop hackathons + Eventbrite India tech
  Workshops:   Unstop workshops-webinars (api.unstop.com) + Eventbrite workshops

NOTE: EventTribe (eventtribe.com) has been REPLACED with:
  - api.unstop.com for competitions/workshops (SSR, confirmed live)
  - eventbrite.com for tech events (SSR, confirmed)
  10times.com renders as JS binary and cannot be scraped without a browser.
"""

import requests
from bs4 import BeautifulSoup
import logging
import re
import time

from careercompass.scrapers import circuit_breaker
from careercompass.scrapers.circuit_breaker import CircuitBreakerSession

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _session():
    s = CircuitBreakerSession()
    s.headers.update(HEADERS)
    return s


# ── Unstop Tech Fests / Competitions ──────────────────────────────────────────
# Confirmed from live fetch: api.unstop.com/all-opportunities renders SSR cards
# Card structure identical to internships — <a href="api.unstop.com/competitions/<slug>">
#   <h2>Title</h2>  Organizer  "N days left"  prize  eligibility tags
# </a>

UNSTOP_TECHFEST_URLS = [
    "https://api.unstop.com/competitions",
    "https://api.unstop.com/hackathons",
    "https://api.unstop.com/all-opportunities",
]


def scrape_unstop_techfests():
    results   = []
    seen_urls = set()
    sess      = _session()

    for page_url in UNSTOP_TECHFEST_URLS:
        if circuit_breaker.is_open(page_url):
            logger.warning("[Unstop Fests] circuit open — stopping remaining URLs")
            break
        try:
            resp = sess.get(page_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            # Cards: anchors pointing to competition/hackathon detail pages
            cards = soup.find_all("a", href=re.compile(
                r"(api\.unstop\.com|unstop\.com)/(competitions|hackathons|college-fests|conferences)/[^/]+$"
            ))
            if not cards:
                cards = [a for a in soup.find_all("a", href=True)
                         if a.find("h2") and any(k in a.get("href","").lower()
                            for k in ("competition","hackathon","fest","conference"))]

            logger.info("[Unstop Fests] {}: {} cards".format(page_url, len(cards)))

            for card in cards:
                h2 = card.find("h2")
                if not h2:
                    continue
                title = h2.get_text(strip=True)
                if not title or len(title) < 3:
                    continue

                href = card["href"]
                url  = href if href.startswith("http") else "https://unstop.com" + href
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                all_texts = [t.strip() for t in card.stripped_strings
                             if t.strip() and t.strip() != title and len(t.strip()) > 1]
                organizer = all_texts[0] if all_texts else ""
                days_left = next((t for t in all_texts if "day" in t.lower() or "left" in t.lower()), "Open")
                prize     = next((t for t in all_texts if re.search(r"[Rs$]\s*[\d,]+|prize|reward", t, re.I)), "")
                eligibility = [t for t in all_texts if any(k in t.lower() for k in
                               ("undergraduate","postgraduate","engineering","school","open"))]

                # Detect type from URL
                kind = "hackathon" if "hackathon" in url.lower() else "techfest"

                results.append({
                    "title":     title,
                    "organizer": organizer,
                    "url":       url,
                    "date":      days_left,
                    "location":  "Online" if not eligibility else "India",
                    "prize":     prize or "See listing",
                    "tags":      (eligibility[:2] + ["Competition"]) if eligibility else ["Competition"],
                    "source":    "Unstop",
                    "type":      kind,
                    "mode":      "Online",
                    "image":     "",
                })

            time.sleep(0.4)

        except Exception as e:
            logger.error("[Unstop Fests] {}: {}".format(page_url, e))

    logger.info("[Unstop Fests] Final: {}".format(len(results)))
    return results


# ── Eventbrite India Tech Events ───────────────────────────────────────────────
# Eventbrite's event listing pages are SSR. URL:
#   https://www.eventbrite.com/d/india/technology--events/
# Card structure: <article> or <li> with h3 title, date, location.

def scrape_eventbrite_techfests():
    results   = []
    seen_urls = set()
    sess      = _session()

    urls = [
        "https://www.eventbrite.com/d/india/technology--events/",
        "https://www.eventbrite.com/d/india/tech--hackathon/",
        "https://www.eventbrite.com/d/india/software-development--events/",
    ]

    for url in urls:
        if circuit_breaker.is_open(url):
            logger.warning("[Eventbrite] circuit open — stopping remaining URLs")
            break
        try:
            resp = sess.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            # Eventbrite event cards
            cards = (soup.select("article[data-event-id]")
                     or soup.select("[class*='search-event-card']")
                     or soup.select("[class*='EventCard']")
                     or soup.select("li article"))

            logger.info("[Eventbrite] {}: {} cards".format(url, len(cards)))

            for card in cards:
                title_el = (card.select_one("h3") or card.select_one("h2")
                            or card.select_one("[class*='event-title']")
                            or card.select_one("[class*='title']"))
                if not title_el:
                    continue
                title = title_el.get_text(strip=True)
                if not title or len(title) < 3:
                    continue

                link_el  = card.select_one("a[href*='eventbrite.com/e/']") or card.select_one("a[href]")
                event_url = link_el["href"] if link_el else url
                if not event_url.startswith("http"):
                    event_url = "https://www.eventbrite.com" + event_url
                if event_url in seen_urls:
                    continue
                seen_urls.add(event_url)

                date_el  = card.select_one("time, [class*='date'], [class*='time']")
                date_str = date_el.get_text(strip=True) if date_el else "Upcoming"

                loc_el   = card.select_one("[class*='location'], [class*='venue']")
                location = loc_el.get_text(strip=True) if loc_el else "India"

                img_el   = card.select_one("img[src]")
                image    = img_el["src"] if img_el else ""

                results.append({
                    "title":     title,
                    "organizer": "",
                    "url":       event_url,
                    "date":      date_str,
                    "location":  location,
                    "prize":     "",
                    "tags":      ["Tech Event", "India"],
                    "source":    "Eventbrite",
                    "type":      "techfest",
                    "mode":      "In-Person" if "online" not in location.lower() else "Online",
                    "image":     image,
                })

            time.sleep(0.5)

        except Exception as e:
            logger.error("[Eventbrite] {}: {}".format(url, e))

    logger.info("[Eventbrite Fests] Final: {}".format(len(results)))
    return results


# ── Unstop Workshops & Webinars ────────────────────────────────────────────────
# api.unstop.com/workshops-webinars — same SSR card pattern

UNSTOP_WORKSHOP_URLS = [
    "https://api.unstop.com/workshops-webinars",
    "https://api.unstop.com/conferences",
]


def scrape_unstop_workshops():
    results   = []
    seen_urls = set()
    sess      = _session()

    for page_url in UNSTOP_WORKSHOP_URLS:
        if circuit_breaker.is_open(page_url):
            logger.warning("[Unstop Workshops] circuit open — stopping remaining URLs")
            break
        try:
            resp = sess.get(page_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            cards = soup.find_all("a", href=re.compile(
                r"(api\.unstop\.com|unstop\.com)/(workshops-webinars|conferences|college-fests)/[^/]+$"
            ))
            if not cards:
                cards = [a for a in soup.find_all("a", href=True)
                         if a.find("h2") and any(k in a.get("href","").lower()
                            for k in ("workshop","webinar","conference","seminar"))]

            logger.info("[Unstop Workshops] {}: {} cards".format(page_url, len(cards)))

            for card in cards:
                h2 = card.find("h2")
                if not h2:
                    continue
                title = h2.get_text(strip=True)
                if not title or len(title) < 3:
                    continue

                href = card["href"]
                url  = href if href.startswith("http") else "https://unstop.com" + href
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                all_texts = [t.strip() for t in card.stripped_strings
                             if t.strip() and t.strip() != title and len(t.strip()) > 1]
                organizer = all_texts[0] if all_texts else ""
                days_left = next((t for t in all_texts if "day" in t.lower() or "left" in t.lower()), "Upcoming")
                fee_text  = next((t for t in all_texts if re.search(r"free|fee|Rs|Rs\s*\d+", t, re.I)), "Free")

                results.append({
                    "title":     title,
                    "organizer": organizer,
                    "url":       url,
                    "date":      days_left,
                    "location":  "Online",
                    "fee":       fee_text,
                    "tags":      ["Workshop", "Online"],
                    "source":    "Unstop",
                    "type":      "workshop",
                    "mode":      "Online",
                    "image":     "",
                })

            time.sleep(0.4)

        except Exception as e:
            logger.error("[Unstop Workshops] {}: {}".format(page_url, e))

    logger.info("[Unstop Workshops] Final: {}".format(len(results)))
    return results


# ── Eventbrite Workshops ───────────────────────────────────────────────────────
def scrape_eventbrite_workshops():
    results   = []
    seen_urls = set()
    sess      = _session()

    urls = [
        "https://www.eventbrite.com/d/india/coding--workshop/",
        "https://www.eventbrite.com/d/online/programming--workshop/",
    ]

    for url in urls:
        if circuit_breaker.is_open(url):
            logger.warning("[Eventbrite WS] circuit open — stopping remaining URLs")
            break
        try:
            resp = sess.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            cards = (soup.select("article[data-event-id]")
                     or soup.select("[class*='search-event-card']")
                     or soup.select("li article"))

            logger.info("[Eventbrite WS] {}: {} cards".format(url, len(cards)))

            for card in cards:
                title_el = card.select_one("h3, h2, [class*='title']")
                if not title_el:
                    continue
                title = title_el.get_text(strip=True)
                if not title or len(title) < 3:
                    continue

                link_el   = card.select_one("a[href*='eventbrite.com/e/']") or card.select_one("a")
                event_url = link_el["href"] if link_el else url
                if not event_url.startswith("http"):
                    event_url = "https://www.eventbrite.com" + event_url
                if event_url in seen_urls:
                    continue
                seen_urls.add(event_url)

                date_el  = card.select_one("time, [class*='date']")
                date_str = date_el.get_text(strip=True) if date_el else "Upcoming"

                price_el = card.select_one("[class*='price'], [class*='ticket']")
                fee      = price_el.get_text(strip=True) if price_el else "Free"

                results.append({
                    "title":     title,
                    "organizer": "",
                    "url":       event_url,
                    "date":      date_str,
                    "location":  "Online / India",
                    "fee":       fee,
                    "tags":      ["Workshop", "Coding"],
                    "source":    "Eventbrite",
                    "type":      "workshop",
                    "mode":      "Online",
                    "image":     "",
                })

            time.sleep(0.5)

        except Exception as e:
            logger.error("[Eventbrite WS] {}: {}".format(url, e))

    logger.info("[Eventbrite Workshops] Final: {}".format(len(results)))
    return results


# ── Aggregators ────────────────────────────────────────────────────────────────
def get_all_techfests():
    all_results = []
    seen_urls   = set()
    seen_titles = set()

    for fn in [scrape_unstop_techfests, scrape_eventbrite_techfests]:
        try:
            items = fn()
            for item in items:
                url   = item.get("url","").strip().rstrip("/")
                title = item.get("title","").lower().strip()
                if (url and url in seen_urls) or (title in seen_titles) or len(title) < 3:
                    continue
                if url:
                    seen_urls.add(url)
                seen_titles.add(title)
                all_results.append(item)
        except Exception as e:
            logger.error("[Techfests] {}: {}".format(fn.__name__, e))

    logger.info("[Techfests] Aggregated: {}".format(len(all_results)))
    return all_results


def get_all_workshops():
    all_results = []
    seen_urls   = set()
    seen_titles = set()

    for fn in [scrape_eventbrite_workshops]:#scrape_unstop_workshops, 
        try:
            items = fn()
            for item in items:
                url   = item.get("url","").strip().rstrip("/")
                title = item.get("title","").lower().strip()
                if (url and url in seen_urls) or (title in seen_titles) or len(title) < 3:
                    continue
                if url:
                    seen_urls.add(url)
                seen_titles.add(title)
                all_results.append(item)
        except Exception as e:
            logger.error("[Workshops] {}: {}".format(fn.__name__, e))

    logger.info("[Workshops] Aggregated: {}".format(len(all_results)))
    return all_results