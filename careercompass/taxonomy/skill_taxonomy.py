"""
taxonomy/skill_taxonomy.py — Canonical skill vocabulary + aliases.

The canonical skill list is NOT duplicated here. It's read from
Config.SLUG_ICONS (skill entries) and Config.ROLE_SLUGS at import time,
so there is exactly one place — config.py — where "what skills/roles
exist" is defined. This file only adds ALIASES.
"""
from careercompass.config import Config

CANONICAL_SKILLS: frozenset = frozenset(
    slug for slug in Config.SLUG_ICONS.keys() if slug not in Config.ROLE_SLUGS
)

SKILL_ALIASES: dict = {
    "javascript": ["js", "es6", "ecmascript", "vanilla js"],
    "typescript": ["ts"],
    "python": ["py", "python3"],
    "nodejs": ["node", "node.js"],
    "react": ["reactjs", "react.js"],
    "vue": ["vuejs", "vue.js"],
    "angular": ["angularjs"],
    "cpp": ["c++"],
    "c-sharp": ["c#", "csharp", "dotnet", ".net"],
    "golang": ["go"],
    "docker": ["containerization", "containers"],
    "kubernetes": ["k8s"],
    "aws": ["amazon web services"],
    "sql": ["mysql", "postgresql", "postgres", "database", "rdbms"],
    "mongodb": ["mongo", "nosql"],
    "git-github": ["git", "github", "version control"],
    "machine-learning": ["ml", "deep learning", "dl"],
    "system-design": ["distributed systems", "system architecture"],
    "datastructures-and-algorithms": ["dsa", "data structures", "algorithms"],
    "html": ["html5"],
    "css": ["css3", "sass", "scss"],
    "computer-science": ["cs", "comp sci"],
    "nextjs": ["next.js", "next"],
    "spring-boot": ["spring"],
    "graphql": ["graph ql"],
    "react-native": ["reactnative"],
    "blockchain": ["web3", "solidity", "smart contracts"],
    "cyber-security": ["security", "ethical hacking", "infosec"],
}

ROLE_ALIASES: dict = {
    "frontend": ["front-end", "front end", "ui developer"],
    "backend": ["back-end", "back end"],
    "full-stack": ["fullstack", "full stack"],
    "devops": ["dev ops", "sre", "site reliability", "ci/cd"],
    "data-analyst": ["data analytics", "business analyst"],
    "ai-engineer": ["ai/ml engineer", "genai engineer", "ai", "ai/ml"],
    "data-engineer": ["data pipeline engineer", "data eng"],
    "ux-design": ["ui/ux", "product design", "figma"],
    "cyber-security": ["cybersecurity", "infosec", "security engineer"],
    "android": ["kotlin", "android developer"],
    "ios": ["swift", "ios developer"],
}


def all_slugs_with_aliases() -> dict:
    """Returns {slug: [slug, *aliases]} for the full matching vocabulary.

    Iterates the UNION of Config.SLUG_ICONS keys and Config.ROLE_SLUGS,
    and combines BOTH SKILL_ALIASES and ROLE_ALIASES for every slug —
    rather than partitioning slugs into two mutually-exclusive buckets
    and only consulting one alias dict per bucket. Several slugs (e.g.
    'machine-learning', 'cyber-security') are dual-classified: present
    in both SLUG_ICONS and ROLE_SLUGS. The old partitioned approach
    silently dropped SKILL_ALIASES for any such slug, since it only
    matched CANONICAL_SKILLS (= SLUG_ICONS minus ROLE_SLUGS) against
    SKILL_ALIASES. Concretely verified: 'ml' should have exact-matched
    machine-learning via its SKILL_ALIASES entry, but never got the
    chance to — it fell through to fuzzy matching and landed on 'html'
    instead. Combining both dicts per-slug, unconditionally, removes
    this entire class of silent alias loss.
    """
    all_slugs = set(Config.SLUG_ICONS.keys()) | set(Config.ROLE_SLUGS)
    table = {}
    for slug in all_slugs:
        aliases = [slug] + SKILL_ALIASES.get(slug, []) + ROLE_ALIASES.get(slug, [])
        table[slug] = aliases
    return table