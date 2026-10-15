"""
hackathons.py — Scraper for all hackathon sources
Uses the robust multi-page API approach from HackRadar for Devpost,
and improved HTML parsing strategies for Devfolio and MLH.
Sources: Devpost (full paginated API), Devfolio (REST API + HTML fallback), MLH (JSON blob + HTML cards)
"""

import requests
from bs4 import BeautifulSoup
import json
import logging
import re
import time
from datetime import datetime

from careercompass.scrapers import circuit_breaker
from careercompass.scrapers.circuit_breaker import CircuitBreakerSession

logger = logging.getLogger(__name__)

TODAY = datetime.now().date()


# ─── Shared session factory ────────────────────────────────────────────────────
def _devpost_session():
    """Session tuned for Devpost's API — needs XHR headers."""
    s = CircuitBreakerSession()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://devpost.com/hackathons",
        "X-Requested-With": "XMLHttpRequest",
    })
    return s


def _html_session():
    """Session for plain HTML scraping."""
    s = CircuitBreakerSession()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


# ─── Devpost helpers ───────────────────────────────────────────────────────────
'''def _parse_prize(raw):
    """Return a cleaned prize string, or 'See website'."""
    if not raw:
        return "See website"
    raw = str(raw).strip()
    if raw in ("", "0", "null", "None"):
        return "See website"
    if raw[0] in ("$", "€", "£", "₹"):
        return raw
    try:
        v = float(raw)
        return "See website" if v == 0 else f"${v:,.0f}"
    except ValueError:
        return raw'''

def _parse_prize(raw):
    """Return a cleaned prize string, stripping any HTML tags.
    Returns 'No prize money' when the value is zero/empty."""
    if not raw:
        return "No prize money"
    raw = str(raw).strip()

    # Strip HTML tags (e.g. "$<span data-currency-value>3,000</span>" → "$3,000")
    raw = re.sub(r"<[^>]+>", "", raw).strip()

    if raw in ("", "0", "null", "None"):
        return "No prize money"

    # Already has a currency symbol — return as-is
    if raw and raw[0] in ("$", "€", "£", "₹"):
        return raw

    # Plain numeric value
    try:
        v = float(raw.replace(",", ""))
        return "No prize money" if v == 0 else f"${v:,.0f}"
    except ValueError:
        return raw if raw else "No prize money"


def _normalize_location(loc_obj, fallback="Online"):
    """Parse Devpost's displayed_location dict into a clean string."""
    if not loc_obj:
        return fallback
    if isinstance(loc_obj, str):
        return loc_obj or fallback
    loc     = loc_obj.get("location", "") or ""
    country = loc_obj.get("country_code", "") or ""
    if loc and country and country.upper() not in loc.upper():
        return f"{loc}, {country.upper()}"
    return loc or fallback


def _dates_from_period(period_str):
    """Split 'Apr 01, 2026 – May 01, 2026' → ('Apr 01, 2026', 'May 01, 2026')."""
    if not period_str:
        return "TBA", "TBA"
    parts = re.split(r"\s*[–\-]\s*", period_str, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return period_str.strip(), "TBA"


# ─── Devpost — full paginated API ─────────────────────────────────────────────
def scrape_devpost():
    """
    Uses Devpost's undocumented JSON API with:
      - status[] bracket notation  → fetches BOTH open + upcoming
      - per_page=24                → 24 results per page (default is only 9)
      - Iterates ALL pages until API returns empty
      - Extracts: location, dates, prize, themes, registrations, thumbnail
    """
    base_url = "https://devpost.com/api/hackathons"
    sess     = _devpost_session()
    results  = []
    seen_ids = set()
    page     = 1

    while True:
        # CRITICAL: pass status as a list — requests encodes as status[]=open&status[]=upcoming
        params = [
            ("status[]", "open"),
            ("status[]", "upcoming"),
            ("page",     page),
            ("per_page", 24),
            ("order_by", "recently-added"),
        ]
        try:
            resp = sess.get(base_url, params=params, timeout=12)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.JSONDecodeError:
            logger.warning(f"[Devpost] Non-JSON response on page {page}, stopping.")
            break
        except Exception as e:
            logger.error(f"[Devpost] Error on page {page}: {e}")
            break

        hackathons = data.get("hackathons", [])
        if not hackathons:
            logger.info(f"[Devpost] Empty page {page} — done. Total: {len(results)}")
            break

        for h in hackathons:
            hid = str(h.get("id", ""))
            if hid in seen_ids:
                continue
            seen_ids.add(hid)

            # ── Location ──────────────────────────────────────────────────
            loc_obj  = h.get("displayed_location") or {}
            location = _normalize_location(loc_obj)
            city     = h.get("city") or ""
            state    = h.get("state") or ""
            country  = h.get("country") or ""
            if not location or location == "Online":
                parts = [p for p in [city, state, country] if p]
                location = ", ".join(parts) if parts else "Online"

            # ── Dates ─────────────────────────────────────────────────────
            period          = h.get("submission_period_dates") or ""
            start_d, end_d  = _dates_from_period(period)

            # ── Themes → tags ──────────────────────────────────────────────
            themes = [t.get("name", "") for t in (h.get("themes") or []) if t.get("name")]
            if not themes:
                themes = ["Hackathon"]

            # ── Prize ──────────────────────────────────────────────────────
            prize = _parse_prize(h.get("prize_amount"))

            # ── Mode ──────────────────────────────────────────────────────
            mode = "Online" if location.lower() in ("online", "") else "In-Person"

            results.append({
                "title":        h.get("title") or "Untitled",
                "url":          h.get("url") or "https://devpost.com/hackathons",
                "image":        h.get("thumbnail_url") or "",
                "prize":        prize,
                "date":         period or start_d,
                "start_date":   start_d,
                "end_date":     end_d,
                "tags":         themes,
                "participants": str(h.get("registrations_count") or ""),
                "source":       "Devpost",
                "type":         "hackathon",
                "mode":         mode,
                "location":     location,
                "organizer":    h.get("organization_name") or "Devpost",
                "status":       h.get("open_state") or "open",
                "description":  h.get("description") or "",
            })

        # ── Pagination: stop when we've consumed all pages ─────────────────
        meta         = data.get("meta", {})
        total_count  = meta.get("total_count", 0)
        per_page_val = meta.get("per_page", 24)
        cur_page     = meta.get("page", page)

        if total_count and per_page_val:
            total_pages = (total_count + per_page_val - 1) // per_page_val
            if cur_page >= total_pages:
                logger.info(f"[Devpost] All {total_pages} pages done. Total: {len(results)}")
                break
        elif len(hackathons) < 24:
            logger.info(f"[Devpost] Last page detected at page {page}. Total: {len(results)}")
            break

        page += 1
        time.sleep(0.3)   # be polite to the API

    logger.info(f"[Devpost] Final count: {len(results)}")
    return results


# ─── Devfolio — REST API + HTML fallback ──────────────────────────────────────
def scrape_devfolio():
    """
    Tries Devfolio's public REST API first.
    Falls back to HTML scraping if the API returns nothing.
    Paginates through all API pages.
    """
    results  = []
    seen_ids = set()

    # ── Strategy 1: REST API ───────────────────────────────────────────────
    api_urls = [
        "https://api.devfolio.co/api/hackathons?open_for_applications=true&page=1&per_page=20",
        "https://api.devfolio.co/api/hackathons?upcoming=true&page=1&per_page=20",
    ]
    sess = CircuitBreakerSession()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
        "Origin": "https://devfolio.co",
        "Referer": "https://devfolio.co/",
    })

    for api_url in api_urls:
        if circuit_breaker.is_open(api_url):
            logger.warning("[Devfolio API] circuit open — stopping remaining API URLs")
            break
        page = 1
        while True:
            url = re.sub(r"page=\d+", f"page={page}", api_url)
            try:
                resp = sess.get(url, timeout=12)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.debug(f"[Devfolio API] page {page} error: {e}")
                break

            # API returns either {"results": [...]} or a list directly
            items = data.get("results", data if isinstance(data, list) else [])
            if not items:
                break

            for h in items:
                hid = h.get("id") or h.get("slug") or h.get("name", "")
                if str(hid) in seen_ids:
                    continue
                seen_ids.add(str(hid))

                slug   = h.get("slug", "")
                title  = h.get("name") or h.get("title") or ""
                if not title:
                    continue
                desc   = h.get("tagline") or h.get("description") or ""
                starts = h.get("starts_at") or h.get("start_date") or ""
                ends   = h.get("ends_at") or h.get("end_date") or ""
                prize  = h.get("prize_pool") or h.get("prize") or ""
                cover  = h.get("cover_image") or h.get("banner") or ""
                online = h.get("is_online", True)
                city   = h.get("city") or ""
                mode   = "Online" if online else f"In-Person{(' • ' + city) if city else ''}"

                # Format dates
                def fmt_date(d):
                    if not d:
                        return "TBD"
                    try:
                        dt = datetime.fromisoformat(str(d).replace("Z", "+00:00"))
                        return dt.strftime("%b %d, %Y")
                    except Exception:
                        return str(d)[:10]

                start_str = fmt_date(starts)
                end_str   = fmt_date(ends)
                date_str  = f"{start_str} – {end_str}" if ends else start_str

                # Format prize
                if isinstance(prize, (int, float)) and prize:
                    prize_str = f"₹{prize:,.0f}"
                elif prize:
                    prize_str = str(prize)
                else:
                    prize_str = "See website"

                results.append({
                    "title":        title,
                    "url":          f"https://devfolio.co/hackathons/{slug}" if slug else "https://devfolio.co/hackathons",
                    "image":        cover,
                    "prize":        prize_str,
                    "date":         date_str,
                    "start_date":   start_str,
                    "end_date":     end_str,
                    "tags":         ["Hackathon", "Devfolio"],
                    "participants": "",
                    "source":       "Devfolio",
                    "type":         "hackathon",
                    "mode":         mode,
                    "location":     city or "Online",
                    "organizer":    "",
                    "status":       "open",
                    "description":  desc,
                })

            # Paginate
            total     = data.get("count") if isinstance(data, dict) else 0
            per_page  = 20
            if total and (page * per_page) >= total:
                break
            elif len(items) < per_page:
                break
            page += 1
            time.sleep(0.3)

    # ── Strategy 2: HTML fallback if API gave nothing ──────────────────────
    if not results:
        logger.info("[Devfolio] API returned nothing, trying HTML scrape...")
        try:
            resp = _html_session().get("https://devfolio.co/hackathons", timeout=15)
            soup = BeautifulSoup(resp.text, "lxml")

            # Try embedded JSON in Next.js __NEXT_DATA__
            next_data = soup.find("script", id="__NEXT_DATA__")
            if next_data and next_data.string:
                try:
                    nd   = json.loads(next_data.string)
                    hacks = (nd.get("props", {})
                               .get("pageProps", {})
                               .get("hackathons", []))
                    for h in hacks:
                        slug  = h.get("slug", "")
                        title = h.get("name") or h.get("title") or ""
                        if not title:
                            continue
                        results.append({
                            "title":        title,
                            "url":          f"https://devfolio.co/hackathons/{slug}" if slug else "https://devfolio.co/hackathons",
                            "image":        h.get("cover_image") or "",
                            "prize":        "See website",
                            "date":         "Ongoing",
                            "start_date":   "TBD",
                            "end_date":     "TBD",
                            "tags":         ["Hackathon"],
                            "participants": "",
                            "source":       "Devfolio",
                            "type":         "hackathon",
                            "mode":         "Online",
                            "location":     "Online",
                            "organizer":    "",
                            "status":       "open",
                            "description":  h.get("tagline") or "",
                        })
                    logger.info(f"[Devfolio] __NEXT_DATA__ gave {len(results)} hackathons")
                except Exception as e:
                    logger.debug(f"[Devfolio] __NEXT_DATA__ parse error: {e}")

            # Fallback to generic card parsing
            if not results:
                cards = soup.select(".HackathonCard, [class*='hackathon-card'], [class*='HackCard'], article")
                for card in cards[:20]:
                    title_el = card.select_one("h2, h3, [class*='title']")
                    link_el  = card.select_one("a[href]")
                    if not title_el:
                        continue
                    title = title_el.get_text(strip=True)
                    link  = link_el["href"] if link_el else "https://devfolio.co/hackathons"
                    if not link.startswith("http"):
                        link = "https://devfolio.co" + link
                    results.append({
                        "title":        title,
                        "url":          link,
                        "image":        "",
                        "prize":        "See website",
                        "date":         "Ongoing",
                        "start_date":   "TBD",
                        "end_date":     "TBD",
                        "tags":         ["Hackathon"],
                        "participants": "",
                        "source":       "Devfolio",
                        "type":         "hackathon",
                        "mode":         "Online",
                        "location":     "Online",
                        "organizer":    "",
                        "status":       "open",
                        "description":  "",
                    })
        except Exception as e:
            logger.error(f"[Devfolio] HTML fallback error: {e}")

    logger.info(f"[Devfolio] Final count: {len(results)}")
    return results


# ─── MLH — JSON blob + HTML card strategies ────────────────────────────────────
def scrape_mlh():
    """
    Two strategies (same approach as HackRadar):
      1. Look for window.__events JSON blob embedded in <script> tags
      2. Parse HTML event cards with multi-format date parsing
    Filters out events that have already ended.
    Tries both 2025 and 2026 season URLs.
    """
    results = []
    seen_names = set()

    season_urls = [
        "https://mlh.io/seasons/2026/events",
        "https://mlh.io/seasons/2025/events",
    ]

    sess = _html_session()

    for url in season_urls:
        if circuit_breaker.is_open(url):
            logger.warning("[MLH] circuit open — stopping remaining season URLs")
            break
        try:
            resp = sess.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # ── Strategy 1: JSON blob in <script> tags ─────────────────────
            json_found = False
            for script in soup.find_all("script"):
                text = script.string or ""
                match = re.search(r"window\.__events\s*=\s*(\[.*?\])\s*;", text, re.DOTALL)
                if match:
                    try:
                        events = json.loads(match.group(1))
                        for ev in events:
                            name     = ev.get("name") or ev.get("title") or ""
                            if not name or name.lower() in seen_names:
                                continue
                            url_link = ev.get("url") or ev.get("link") or "https://mlh.io"
                            start    = ev.get("start_date") or ev.get("startDate") or "TBA"
                            end      = ev.get("end_date")   or ev.get("endDate")   or "TBA"
                            location = ev.get("location") or "USA"
                            image    = ev.get("image") or ev.get("logo") or ""

                            # Skip past events
                            try:
                                end_dt   = datetime.strptime(end,   "%Y-%m-%d").date()
                                start_dt = datetime.strptime(start, "%Y-%m-%d").date()
                                if end_dt < TODAY:
                                    continue
                                status   = "open" if start_dt <= TODAY else "upcoming"
                                date_str = f"{start_dt.strftime('%b %d')} – {end_dt.strftime('%b %d, %Y')}"
                            except Exception:
                                status   = "upcoming"
                                date_str = f"{start} – {end}"

                            seen_names.add(name.lower())
                            results.append({
                                "title":        name,
                                "url":          url_link,
                                "image":        image,
                                "prize":        "MLH swag + partner prizes",
                                "date":         date_str,
                                "start_date":   start,
                                "end_date":     end,
                                "tags":         ["MLH", "Collegiate", "In-Person"],
                                "participants": "",
                                "source":       "MLH",
                                "type":         "hackathon",
                                "mode":         "In-Person",
                                "location":     location,
                                "organizer":    "MLH",
                                "status":       status,
                                "description":  f"Official MLH member event. {date_str}.",
                            })
                        json_found = True
                        logger.info(f"[MLH] JSON blob strategy: {len(results)} events from {url}")
                    except Exception as e:
                        logger.debug(f"[MLH] JSON blob parse error: {e}")

            if json_found and results:
                continue   # move to next season URL if needed

            # ── Strategy 2: HTML event cards ───────────────────────────────
            cards = (
                soup.find_all("div", class_=re.compile(r"\bevent\b", re.I)) or
                soup.find_all("a",   class_=re.compile(r"\bevent\b", re.I))
            )
            logger.info(f"[MLH] HTML card strategy: found {len(cards)} cards from {url}")

            for card in cards:
                # Name
                name_el = (
                    card.find(class_=re.compile(r"event.?name|title", re.I)) or
                    card.find("h3") or card.find("h4") or card.find("strong")
                )
                name = name_el.get_text(strip=True) if name_el else ""
                if not name or len(name) < 3 or name.lower() in seen_names:
                    continue

                # Link
                link_el  = card if card.name == "a" else card.find("a", href=True)
                url_link = ""
                if link_el:
                    href = link_el.get("href", "")
                    if href.startswith("http"):
                        url_link = href
                    elif href.startswith("/"):
                        url_link = "https://mlh.io" + href

                # Date string
                date_el  = (card.find(class_=re.compile(r"date|time", re.I))
                             or card.find("time"))
                date_str = date_el.get_text(strip=True) if date_el else "TBA"

                # Location
                loc_el   = card.find(class_=re.compile(r"location|city|place", re.I))
                location = loc_el.get_text(strip=True) if loc_el else "USA"

                # Image
                img_el = card.find("img")
                image  = img_el.get("src", "") if img_el else ""

                # Parse start/end from the date string
                # Handles: "Apr 4-6, 2026", "March 28 - 29, 2026", "Apr 4 - May 1, 2026"
                status     = "upcoming"
                start_date = "TBA"
                end_date   = "TBA"
                try:
                    year_m = re.search(r"\b(202\d)\b", date_str)
                    year   = int(year_m.group(1)) if year_m else TODAY.year

                    # Pattern: "MonthA day1 - MonthB day2, year"
                    cross_m = re.search(
                        r"([A-Za-z]+)\s+(\d+)\s*[-–]\s*([A-Za-z]+)\s+(\d+)",
                        date_str
                    )
                    same_m = re.search(
                        r"([A-Za-z]+)\s+(\d+)\s*[-–]\s*(\d+)",
                        date_str
                    )

                    def try_parse(month_str, day, yr):
                        for fmt in ("%b %d %Y", "%B %d %Y"):
                            try:
                                return datetime.strptime(f"{month_str} {day} {yr}", fmt).date()
                            except ValueError:
                                continue
                        return None

                    if cross_m:
                        start_dt = try_parse(cross_m.group(1), int(cross_m.group(2)), year)
                        end_dt   = try_parse(cross_m.group(3), int(cross_m.group(4)), year)
                    elif same_m:
                        start_dt = try_parse(same_m.group(1), int(same_m.group(2)), year)
                        end_dt   = try_parse(same_m.group(1), int(same_m.group(3)), year)
                    else:
                        start_dt = end_dt = None

                    if start_dt and end_dt:
                        if end_dt < TODAY:
                            continue   # already finished
                        status     = "open"     if start_dt <= TODAY else "upcoming"
                        start_date = start_dt.strftime("%b %d, %Y")
                        end_date   = end_dt.strftime("%b %d, %Y")
                except Exception:
                    pass

                seen_names.add(name.lower())
                results.append({
                    "title":        name,
                    "url":          url_link or url,
                    "image":        image,
                    "prize":        "MLH swag + partner prizes",
                    "date":         date_str,
                    "start_date":   start_date,
                    "end_date":     end_date,
                    "tags":         ["MLH", "Collegiate", "In-Person"],
                    "participants": "",
                    "source":       "MLH",
                    "type":         "hackathon",
                    "mode":         "In-Person",
                    "location":     location,
                    "organizer":    "MLH",
                    "status":       status,
                    "description":  f"Official MLH member hackathon. {date_str}.",
                })

        except Exception as e:
            logger.error(f"[MLH] Failed to fetch {url}: {e}")

    logger.info(f"[MLH] Final count: {len(results)}")
    return results


# ─── Aggregator ────────────────────────────────────────────────────────────────
def get_all_hackathons():
    """
    Runs all three scrapers, de-duplicates by URL first, then by title.
    Returns a merged, deduplicated list.
    """
    all_results  = []
    seen_urls    = set()
    seen_titles  = set()

    for fn in [scrape_devpost, scrape_devfolio, scrape_mlh]:
        try:
            items = fn()
            for item in items:
                url   = item.get("url", "").strip().rstrip("/")
                title = item.get("title", "").lower().strip()

                # De-duplicate by URL first (most reliable), then title
                if url and url != "#" and url in seen_urls:
                    continue
                if title in seen_titles or len(title) < 3:
                    continue

                if url and url != "#":
                    seen_urls.add(url)
                seen_titles.add(title)
                all_results.append(item)
        except Exception as e:
            logger.error(f"[Hackathons] Error in {fn.__name__}: {e}")

    logger.info(f"[Hackathons] Aggregated total: {len(all_results)}")
    return all_results