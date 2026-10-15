"""
internships.py — Multi-source internship scraper
Sources:
  1. Internshala — SSR paginated pages, confirmed live selectors
  2. Unstop      — api.unstop.com SSR listing pages, confirmed live
  3. Wellfound   — /role/r/<slug>?page=N SSR pages, confirmed live
  4. LinkedIn    — public search (bonus, graceful fallback)
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


# Confirmed selectors from live fetch of internshala.com/internships/computer-science-internship/
# Cards:    <h2><a href="/internship/detail/<slug>">Title</a></h2>
# Company:  .company-name a  /  .company_name a
# Stipend:  .stipend            -> "Rs 5,000 - 7,500 /month" / "Unpaid"
# Location: .location_link      -> "Kolkata" / "Work from home"
# Duration: .duration           -> "6 Months"
# Skills:   [class*=round_tabs] -> individual skill spans
# Next page: a[href*="page-N"]  -> /page-2/ through /page-15/

INTERNSHALA_CATEGORIES = [
    ("computer-science",         "CS"),
    ("software-development",     "SWE"),
    ("web-development",          "Web Dev"),
    ("data-science",             "Data Science"),
    ("artificial-intelligence",  "AI/ML"),
    ("machine-learning",         "ML"),
    ("python",                   "Python"),
    ("android-app-development",  "Android"),
]


_SKILL_CLASS_RE = re.compile(r"round_tabs|skill_tag|skill", re.IGNORECASE)


def _card_scope(h2):
    """
    Returns elements belonging to THIS listing only — bounded by document
    position between this h2 and the next one, not by any wrapper class
    name. This replaced an ancestor-climbing heuristic that looked for a
    wrapper class containing 'individual'/'internship'/'card'/'listing'.

    That heuristic broke silently: if none of those keywords appeared in
    any ancestor within 6 levels (e.g. after a site redesign), the loop
    just used whatever the 6th-level ancestor happened to be — often a
    page-wide container. Every listing then hit the same failure and
    landed on the same over-broad scope, so `card.select(...)` returned
    the SAME elements (e.g. a sidebar filter widget's first entries) for
    every single card on the page. Reproduced concretely: this produced
    the identical tag "CSS" on every listing regardless of actual content
    — confirmed against the ancestor-climbing version with a synthetic
    page matching this exact failure mode.

    Bounding by "stuff between this h2 and the next h2 in document order"
    doesn't depend on any class name existing at all, so it can't break
    the same way. It also naturally excludes anything positioned BEFORE
    the first h2 (e.g. a sidebar filter widget), since find_all_next only
    searches forward.
    """
    scope = []
    for el in h2.find_all_next():
        if el.name == "h2":
            break
        scope.append(el)
    return scope


def _find_in_scope(scope, selector_fn):
    """First element in `scope` matching selector_fn(el) -> bool."""
    for el in scope:
        if selector_fn(el):
            return el
    return None


def _skills_in_scope(scope, limit=6):
    """Leaf skill-tag elements only — explicitly skips any element that
    itself contains another matching child, since a wrapper div (e.g.
    class="opportunity_block_skills") also matches the broad `skill`
    substring pattern, and .get_text() on a wrapper concatenates all its
    children's text with no separator (garbled, e.g. "PythonDjangoFlask"
    instead of separate "Python"/"Django"/"Flask" entries)."""
    out = []
    for el in scope:
        cls = " ".join(el.get("class", []))
        if not _SKILL_CLASS_RE.search(cls):
            continue
        if el.find(attrs={"class": _SKILL_CLASS_RE}):
            continue  # wrapper containing its own matching children — skip
        text = el.get_text(strip=True)
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _parse_internshala_card(h2, cat_label):
    link_el = h2.find("a", href=re.compile(r"/internship/detail/"))
    if not link_el:
        return None
    title = h2.get_text(strip=True)
    href  = link_el["href"]
    url   = href if href.startswith("http") else "https://internshala.com" + href

    scope = _card_scope(h2)

    def _sel(class_substr):
        el = _find_in_scope(scope, lambda e: class_substr in " ".join(e.get("class", [])).lower())
        return el.get_text(strip=True) if el else ""

    company  = _sel("company-name") or _sel("company_name") or _sel("company")
    stipend  = _sel("stipend") or "Unpaid"
    location = _sel("location_link") or _sel("location") or _sel("row-1-item")
    duration = _sel("duration")

    skills = _skills_in_scope(scope, limit=6)

    mode = "Remote" if ("work from home" in location.lower() or "remote" in location.lower()) else "On-Site"

    return {
        "title":    title,
        "company":  company,
        "url":      url,
        "location": location or "India",
        "stipend":  stipend,
        "duration": duration,
        "tags":     list(set((skills or []) + [cat_label, "Internship"]))[:6],
        "source":   "Internshala",
        "type":     "internship",
        "mode":     mode,
        "image":    "",
        "date":     "Open",
    }


def scrape_internshala(max_pages=3):
    results   = []
    seen_urls = set()
    sess      = _session()

    for cat_slug, cat_label in INTERNSHALA_CATEGORIES:
        if circuit_breaker.is_open("https://internshala.com"):
            logger.warning("[Internshala] circuit open — stopping remaining categories")
            break
        base = "https://internshala.com/internships/{}-internship".format(cat_slug)

        for page in range(1, max_pages + 1):
            url = base if page == 1 else "{}/page-{}/".format(base, page)
            try:
                resp = sess.get(url, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "lxml")

                h2_tags = [h2 for h2 in soup.find_all("h2")
                           if h2.find("a", href=re.compile(r"/internship/detail/"))]
                if not h2_tags:
                    break

                new_cnt = 0
                for h2 in h2_tags:
                    item = _parse_internshala_card(h2, cat_label)
                    if item and item["url"] not in seen_urls and len(item["title"]) > 2:
                        seen_urls.add(item["url"])
                        results.append(item)
                        new_cnt += 1

                logger.info("[Internshala] {} p{}: {} found, {} new".format(cat_slug, page, len(h2_tags), new_cnt))

                if not soup.select_one("a[href*='page-{}']".format(page + 1)):
                    break
                time.sleep(0.5)

            except Exception as e:
                logger.error("[Internshala] {}: {}".format(url, e))
                break

        time.sleep(0.4)

    logger.info("[Internshala] Final: {}".format(len(results)))
    return results


# Confirmed from live fetch of api.unstop.com/internships/:
# Cards: <a href="https://api.unstop.com/internships/<slug>">
#           <h2>Title</h2>  CompanyName  "X Applied"  "N days left"  eligibility
#        </a>

UNSTOP_INTERNSHIP_URLS = [
    "https://api.unstop.com/internships/",
    "https://api.unstop.com/internship/software-development-internship",
    "https://api.unstop.com/internship/data-science-internship",
    "https://api.unstop.com/internship/applied-ai-internship",
    "https://api.unstop.com/internship/web-development-internship",
    "https://api.unstop.com/internship/machine-learning-internship",
]


def scrape_unstop_internships():
    results   = []
    seen_urls = set()
    sess      = _session()

    for page_url in UNSTOP_INTERNSHIP_URLS:
        if circuit_breaker.is_open(page_url):
            logger.warning("[Unstop internships] circuit open — stopping remaining URLs")
            break
        try:
            resp = sess.get(page_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            cards = soup.find_all("a", href=re.compile(
                r"(api\.unstop\.com|unstop\.com)/internships?/[^/]+$"
            ))
            if not cards:
                cards = [a for a in soup.find_all("a", href=True)
                         if a.find("h2") and ("internship" in a.get("href","").lower()
                                               or "job" in a.get("href","").lower())]

            logger.info("[Unstop] {}: {} cards".format(page_url, len(cards)))

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
                company   = all_texts[0] if all_texts else ""
                days_left = next((t for t in all_texts if "day" in t.lower() or "left" in t.lower()), "Open")
                eligibility = [t for t in all_texts if any(k in t.lower() for k in
                               ("undergraduate","postgraduate","engineering","mba","school"))]

                results.append({
                    "title":    title,
                    "company":  company,
                    "url":      url,
                    "location": "India",
                    "stipend":  "See listing",
                    "duration": "",
                    "tags":     (eligibility[:2] + ["Internship"]) if eligibility else ["Internship"],
                    "source":   "Unstop",
                    "type":     "internship",
                    "mode":     "Remote / On-Site",
                    "image":    "",
                    "date":     days_left,
                })

            time.sleep(0.4)

        except Exception as e:
            logger.error("[Unstop internships] {}: {}".format(page_url, e))

    logger.info("[Unstop internships] Final: {}".format(len(results)))
    return results


# Confirmed from live fetch: wellfound.com/role/r/<slug>?page=N
# Structure: company h2 blocks, job title links = /jobs/<id>-<slug>
# Salary, location, experience in surrounding row text

WELLFOUND_ROLES = [
    ("software-engineer",         "SWE"),
    ("frontend-engineer",         "Frontend"),
    ("backend-engineer",          "Backend"),
    ("full-stack-engineer",       "Full-Stack"),
    ("data-scientist",            "Data Science"),
    ("machine-learning-engineer", "ML"),
    ("mobile-engineer",           "Mobile"),
    ("devops-engineer",           "DevOps"),
    ("react-developer",           "React"),
    ("python-developer",          "Python"),
]


def _parse_wellfound_section(section, label):
    company_el = section.find("h2") or section.find("h3")
    company    = company_el.get_text(strip=True) if company_el else ""

    items = []
    for link in section.find_all("a", href=re.compile(r"/jobs/\d+")):
        title = link.get_text(strip=True)
        if not title or len(title) < 3:
            continue
        href = link["href"]
        url  = href if href.startswith("http") else "https://wellfound.com" + href

        row      = link.find_parent(lambda t: t.name in ("div","li") and len(t.get_text()) > 15) or link.parent
        row_text = row.get_text(" ", strip=True) if row else ""

        sal_m    = re.search(r"([$Rs.]+\s*[\d,\.]+[kKlLmM]?\s*[--]\s*[$Rs.]+\s*[\d,\.]+[kKlLmM]?)", row_text)
        salary   = sal_m.group(1).strip() if sal_m else "Competitive"

        loc_m    = re.search(r"(Remote only|Onsite or remote|Remote)\s*[*.]?\s*([^\n*.]+)", row_text, re.I)
        location = loc_m.group(0).strip()[:60] if loc_m else "Remote"

        exp_m    = re.search(r"(\d+)\s+year", row_text)
        exp      = "{} yrs exp".format(exp_m.group(1)) if exp_m else "Entry Level"

        posted_m = re.search(r"(\d+\s+(?:day|week|month)s?\s+ago|Just now|Today|\d+\s+hour)", row_text, re.I)
        posted   = posted_m.group(1) if posted_m else ""

        items.append({
            "title":    title,
            "company":  company,
            "url":      url,
            "location": location,
            "stipend":  salary,
            "duration": exp,
            "tags":     [label, "Startup", "Full-time"],
            "source":   "Wellfound",
            "type":     "internship",
            "mode":     "Remote" if "remote" in location.lower() else "Hybrid",
            "image":    "",
            "date":     posted or "Open",
        })
    return items


def scrape_wellfound(max_pages=2):
    results   = []
    seen_urls = set()
    sess      = _session()

    for role_slug, label in WELLFOUND_ROLES:
        if circuit_breaker.is_open("https://wellfound.com"):
            logger.warning("[Wellfound] circuit open — stopping remaining roles")
            break
        for page in range(1, max_pages + 1):
            url = "https://wellfound.com/role/r/{}".format(role_slug)
            if page > 1:
                url += "?page={}".format(page)
            try:
                resp = sess.get(url, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "lxml")

                sections = [t for t in soup.find_all(["div","section","li"])
                            if t.find("h2") and t.find("a", href=re.compile(r"/jobs/\d+"))]

                if not sections:
                    # Flat fallback
                    for link in soup.find_all("a", href=re.compile(r"wellfound\.com/jobs/\d+")):
                        title = link.get_text(strip=True)
                        u     = link["href"]
                        if u not in seen_urls and title and len(title) > 2:
                            seen_urls.add(u)
                            results.append({
                                "title": title, "company": "", "url": u,
                                "location": "Remote", "stipend": "Competitive",
                                "duration": "Entry Level", "tags": [label, "Startup"],
                                "source": "Wellfound", "type": "internship",
                                "mode": "Remote", "image": "", "date": "Open",
                            })
                    break

                for sec in sections:
                    for item in _parse_wellfound_section(sec, label):
                        if item["url"] not in seen_urls:
                            seen_urls.add(item["url"])
                            results.append(item)

                logger.info("[Wellfound] {} p{}: {} company blocks".format(role_slug, page, len(sections)))

                # Next page detection
                has_next = bool(soup.find("a", string=re.compile(r"Next", re.I))
                                or soup.find("a", attrs={"rel": "next"}))
                if not has_next:
                    break
                time.sleep(0.5)

            except Exception as e:
                logger.error("[Wellfound] {}: {}".format(url, e))
                break

        time.sleep(0.4)

    logger.info("[Wellfound] Final: {}".format(len(results)))
    return results


def scrape_linkedin_internships():
    results = []
    try:
        url  = "https://www.linkedin.com/jobs/search/?keywords=software+engineering+internship&location=India&f_JT=I"
        sess = _session()
        resp = sess.get(url, timeout=12)
        soup = BeautifulSoup(resp.text, "lxml")
        for card in soup.select(".base-card, .job-search-card")[:20]:
            title_el   = card.select_one("h3.base-search-card__title, h3")
            company_el = card.select_one("h4.base-search-card__subtitle")
            loc_el     = card.select_one(".job-search-card__location, .base-search-card__metadata")
            link_el    = card.select_one("a.base-card__full-link, a[href*='linkedin.com/jobs']")
            date_el    = card.select_one("time")
            if not title_el:
                continue
            results.append({
                "title":    title_el.get_text(strip=True),
                "company":  company_el.get_text(strip=True) if company_el else "",
                "url":      link_el["href"] if link_el else url,
                "location": loc_el.get_text(strip=True) if loc_el else "India",
                "stipend":  "As per company",
                "duration": "Internship",
                "tags":     ["LinkedIn", "Internship"],
                "source":   "LinkedIn",
                "type":     "internship",
                "mode":     "Hybrid",
                "image":    "",
                "date":     date_el.get("datetime","") if date_el else "Recent",
            })
    except Exception as e:
        logger.error("[LinkedIn internships] {}".format(e))
    return results


def get_all_internships():
    all_results = []
    seen_urls   = set()
    seen_titles = set()

    for fn in [scrape_internshala, scrape_unstop_internships, scrape_wellfound, scrape_linkedin_internships]:
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
            logger.error("[Internships] {}: {}".format(fn.__name__, e))

    logger.info("[Internships] Aggregated: {}".format(len(all_results)))
    return all_results