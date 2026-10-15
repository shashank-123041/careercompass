"""
jobs.py — Full-time jobs scraper
Sources:
  1. Internshala fresher-jobs — same SSR structure as internships, confirmed live
  2. Unstop jobs   — api.unstop.com/job/  SSR, same card pattern as internships
  3. Wellfound     — /role/<slug>  (non-remote, same SSR structure, confirmed live)
  4. LinkedIn      — public search bonus
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


# Internshala fresher-jobs: same page structure as /internships/ but under /fresher-jobs/
# and /jobs/ — confirmed SSR with same h2/detail selectors

INTERNSHALA_JOB_CATS = [
    ("computer-science",    "CS"),
    ("software-development","SWE"),
    ("web-development",     "Web Dev"),
    ("data-science",        "Data Science"),
    ("machine-learning",    "ML"),
    ("python",              "Python"),
    ("backend-developer",   "Backend"),
    ("frontend-developer",  "Frontend"),
]


_SKILL_CLASS_RE = re.compile(r"round_tabs|skill_tag|skill", re.IGNORECASE)


def _card_scope(h2):
    """See scrapers/internships.py's _card_scope for the full explanation.
    Bounds a card's content by document position between this h2 and the
    next one, instead of climbing ancestors looking for a wrapper class
    name — resilient to site redesigns that rename those classes, since
    it doesn't depend on any particular class existing at all."""
    scope = []
    for el in h2.find_all_next():
        if el.name == "h2":
            break
        scope.append(el)
    return scope


def _find_in_scope(scope, selector_fn):
    for el in scope:
        if selector_fn(el):
            return el
    return None


def _skills_in_scope(scope, limit=5):
    """Leaf skill-tag elements only — skips wrapper elements that
    themselves contain matching children (see internships.py's version
    for why: a wrapper's .get_text() garbles multiple children's text
    together with no separator)."""
    out = []
    for el in scope:
        cls = " ".join(el.get("class", []))
        if not _SKILL_CLASS_RE.search(cls):
            continue
        if el.find(attrs={"class": _SKILL_CLASS_RE}):
            continue
        text = el.get_text(strip=True)
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _parse_internshala_job_card(h2, cat_label):
    link_el = h2.find("a", href=re.compile(r"/job/detail/|/jobs/detail/"))
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
    salary   = _sel("salary") or _sel("ctc") or "As per company"
    location = _sel("location_link") or _sel("location") or _sel("row-1-item")
    exp      = _sel("experience") or "Fresher"

    skills = _skills_in_scope(scope, limit=5)

    mode = "Remote" if ("work from home" in location.lower() or "remote" in location.lower()) else "On-Site"

    return {
        "title":      title,
        "company":    company,
        "url":        url,
        "location":   location or "India",
        "salary":     salary,
        "experience": exp,
        "tags":       list(set((skills or []) + [cat_label, "Job"]))[:6],
        "source":     "Internshala",
        "type":       "job",
        "mode":       mode,
        "image":      "",
        "date":       "Open",
    }


def scrape_internshala_jobs(max_pages=2):
    results   = []
    seen_urls = set()
    sess      = _session()

    for cat_slug, cat_label in INTERNSHALA_JOB_CATS:
        if circuit_breaker.is_open("https://internshala.com"):
            logger.warning("[IS Jobs] circuit open — stopping remaining categories")
            break
        for base_path in [
            "https://internshala.com/fresher-jobs/{}-jobs".format(cat_slug),
            "https://internshala.com/jobs/{}-jobs".format(cat_slug),
        ]:
            for page in range(1, max_pages + 1):
                url = base_path if page == 1 else "{}/page-{}/".format(base_path, page)
                try:
                    resp = sess.get(url, timeout=15)
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.text, "lxml")

                    h2_tags = [h2 for h2 in soup.find_all("h2")
                               if h2.find("a", href=re.compile(r"/job/detail/|/jobs/detail/"))]
                    if not h2_tags:
                        break

                    new_cnt = 0
                    for h2 in h2_tags:
                        item = _parse_internshala_job_card(h2, cat_label)
                        if item and item["url"] not in seen_urls and len(item["title"]) > 2:
                            seen_urls.add(item["url"])
                            results.append(item)
                            new_cnt += 1

                    logger.info("[IS Jobs] {} p{}: {} found, {} new".format(cat_slug, page, len(h2_tags), new_cnt))

                    if not soup.select_one("a[href*='page-{}']".format(page + 1)):
                        break
                    time.sleep(0.4)

                except Exception as e:
                    logger.error("[IS Jobs] {}: {}".format(url, e))
                    break
            time.sleep(0.3)

    logger.info("[IS Jobs] Final: {}".format(len(results)))
    return results


# Unstop jobs: api.unstop.com/job — same card pattern as internships
UNSTOP_JOB_URLS = [
    "https://api.unstop.com/job",
    "https://api.unstop.com/jobs/software-development-jobs",
    "https://api.unstop.com/jobs/data-science-jobs",
    "https://api.unstop.com/jobs/backend-developer-jobs",
    "https://api.unstop.com/jobs/frontend-developer-jobs",
]


def scrape_unstop_jobs():
    results   = []
    seen_urls = set()
    sess      = _session()

    for page_url in UNSTOP_JOB_URLS:
        if circuit_breaker.is_open(page_url):
            logger.warning("[Unstop jobs] circuit open — stopping remaining URLs")
            break
        try:
            resp = sess.get(page_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            cards = soup.find_all("a", href=re.compile(
                r"(api\.unstop\.com|unstop\.com)/jobs?/[^/]+$"
            ))
            if not cards:
                cards = [a for a in soup.find_all("a", href=True)
                         if a.find("h2") and "job" in a.get("href","").lower()]

            logger.info("[Unstop jobs] {}: {} cards".format(page_url, len(cards)))

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

                all_texts   = [t.strip() for t in card.stripped_strings
                               if t.strip() and t.strip() != title and len(t.strip()) > 1]
                company     = all_texts[0] if all_texts else ""
                days_left   = next((t for t in all_texts if "day" in t.lower() or "left" in t.lower()), "Open")
                eligibility = [t for t in all_texts if any(k in t.lower() for k in
                               ("experienced","fresher","undergraduate","postgraduate","engineering"))]

                results.append({
                    "title":      title,
                    "company":    company,
                    "url":        url,
                    "location":   "India",
                    "salary":     "See listing",
                    "experience": eligibility[0] if eligibility else "Fresher",
                    "tags":       (eligibility[:2] + ["Job"]) if eligibility else ["Job"],
                    "source":     "Unstop",
                    "type":       "job",
                    "mode":       "On-Site",
                    "image":      "",
                    "date":       days_left,
                })

            time.sleep(0.4)

        except Exception as e:
            logger.error("[Unstop jobs] {}: {}".format(page_url, e))

    logger.info("[Unstop jobs] Final: {}".format(len(results)))
    return results


# Wellfound full-time roles (no /r/ remote filter, same SSR structure)
WELLFOUND_JOB_ROLES = [
    ("software-engineer",         "SWE"),
    ("backend-engineer",          "Backend"),
    ("frontend-engineer",         "Frontend"),
    ("full-stack-engineer",       "Full-Stack"),
    ("data-scientist",            "Data Science"),
    ("machine-learning-engineer", "ML"),
    ("data-engineer",             "Data Eng"),
    ("cloud-engineer",            "Cloud"),
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
        location = loc_m.group(0).strip()[:60] if loc_m else "USA/Remote"

        exp_m    = re.search(r"(\d+)\s+year", row_text)
        exp      = "{} yrs exp".format(exp_m.group(1)) if exp_m else "Entry Level"

        posted_m = re.search(r"(\d+\s+(?:day|week|month)s?\s+ago|Just now|Today|\d+\s+hour)", row_text, re.I)
        posted   = posted_m.group(1) if posted_m else ""

        items.append({
            "title":      title,
            "company":    company,
            "url":        url,
            "location":   location,
            "salary":     salary,
            "experience": exp,
            "tags":       [label, "Startup", "Full-time"],
            "source":     "Wellfound",
            "type":       "job",
            "mode":       "Remote" if "remote" in location.lower() else "Hybrid",
            "image":      "",
            "date":       posted or "Open",
        })
    return items


def scrape_wellfound_jobs(max_pages=2):
    results   = []
    seen_urls = set()
    sess      = _session()

    for role_slug, label in WELLFOUND_JOB_ROLES:
        if circuit_breaker.is_open("https://wellfound.com"):
            logger.warning("[Wellfound Jobs] circuit open — stopping remaining roles")
            break
        for page in range(1, max_pages + 1):
            url = "https://wellfound.com/role/{}".format(role_slug)
            if page > 1:
                url += "?page={}".format(page)
            try:
                resp = sess.get(url, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "lxml")

                sections = [t for t in soup.find_all(["div","section","li"])
                            if t.find("h2") and t.find("a", href=re.compile(r"/jobs/\d+"))]

                if not sections:
                    for link in soup.find_all("a", href=re.compile(r"wellfound\.com/jobs/\d+")):
                        title = link.get_text(strip=True)
                        u     = link["href"]
                        if u not in seen_urls and title and len(title) > 2:
                            seen_urls.add(u)
                            results.append({
                                "title": title, "company": "", "url": u,
                                "location": "USA/Remote", "salary": "Competitive",
                                "experience": "Entry Level", "tags": [label, "Startup"],
                                "source": "Wellfound", "type": "job",
                                "mode": "Hybrid", "image": "", "date": "Open",
                            })
                    break

                for sec in sections:
                    for item in _parse_wellfound_section(sec, label):
                        if item["url"] not in seen_urls:
                            seen_urls.add(item["url"])
                            results.append(item)

                logger.info("[Wellfound Jobs] {} p{}: {} blocks".format(role_slug, page, len(sections)))

                has_next = bool(soup.find("a", string=re.compile(r"Next", re.I))
                                or soup.find("a", attrs={"rel": "next"}))
                if not has_next:
                    break
                time.sleep(0.5)

            except Exception as e:
                logger.error("[Wellfound Jobs] {}: {}".format(url, e))
                break

        time.sleep(0.4)

    logger.info("[Wellfound Jobs] Final: {}".format(len(results)))
    return results


def scrape_linkedin_jobs():
    results = []
    try:
        url  = "https://www.linkedin.com/jobs/search/?keywords=software+engineer&location=India&f_JT=F&f_E=2"
        sess = _session()
        resp = sess.get(url, timeout=12)
        soup = BeautifulSoup(resp.text, "lxml")
        for card in soup.select(".base-card, .job-search-card")[:20]:
            title_el   = card.select_one("h3.base-search-card__title, h3")
            company_el = card.select_one("h4.base-search-card__subtitle")
            loc_el     = card.select_one(".job-search-card__location")
            link_el    = card.select_one("a.base-card__full-link, a[href*='linkedin.com/jobs']")
            date_el    = card.select_one("time")
            if not title_el:
                continue
            results.append({
                "title":      title_el.get_text(strip=True),
                "company":    company_el.get_text(strip=True) if company_el else "",
                "url":        link_el["href"] if link_el else url,
                "location":   loc_el.get_text(strip=True) if loc_el else "India",
                "salary":     "As per company",
                "experience": "Entry Level",
                "tags":       ["LinkedIn", "Full-time"],
                "source":     "LinkedIn",
                "type":       "job",
                "mode":       "Hybrid",
                "image":      "",
                "date":       date_el.get("datetime","") if date_el else "Recent",
            })
    except Exception as e:
        logger.error("[LinkedIn jobs] {}".format(e))
    return results


def get_all_jobs():
    all_results = []
    seen_urls   = set()
    seen_titles = set()

    for fn in [scrape_internshala_jobs, scrape_unstop_jobs, scrape_wellfound_jobs, scrape_linkedin_jobs]:
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
            logger.error("[Jobs] {}: {}".format(fn.__name__, e))

    logger.info("[Jobs] Aggregated: {}".format(len(all_results)))
    return all_results