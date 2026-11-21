#!/usr/bin/env python
"""Regenerate ARCHIVE.md from the GitHub API.

Usage:
    GITHUB_TOKEN=<pat> python scripts/gen_archive.py > ARCHIVE.md

The PAT needs read access to the account's repositories. No token -> the
script exits with a message rather than writing a broken file.
"""
import json
import os
import sys
import urllib.request

TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
if not TOKEN:
    sys.exit("set GITHUB_TOKEN to regenerate the archive index")

CONSUMED = {
    "fluidtech", "TMU", "NEVER_LIE", "FactBase", "KNOW-VAULT", "BUS-TRACE",
    "TRACE-PATH", "LAYER-CENSUS-", "GEOMETRY-SCAN", "INNER-VOTE-", "STATE-MIRROR",
    "LOGIT-STREAM", "VARIEANCE-LIES", "ComplexityRouter", "Draft_ponit_harness",
    "NOISE-STORM", "DEAD-CELL", "TWIN-TOWER", "SKIP-RUN-", "POWERLAW-PROBE",
}


def fetch(page: int):
    url = ("https://api.github.com/user/repos?per_page=100&visibility=all"
           f"&sort=updated&page={page}")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def main():
    repos = []
    page = 1
    while True:
        batch = fetch(page)
        if not batch:
            break
        repos += batch
        page += 1

    rows = sorted(
        ({"name": r["name"], "desc": (r.get("description") or "").strip()[:90],
          "url": r["html_url"], "lang": r.get("language")} for r in repos),
        key=lambda r: r["name"].lower())

    out = ["# ARCHIVE — the sibling repositories", "",
           "Generated from the GitHub API by `scripts/gen_archive.py`.", "",
           "## Consumed by VENOM", "",
           "| Repo | Description |", "|---|---|"]
    for r in rows:
        if r["name"] in CONSUMED:
            out.append(f"| [`{r['name']}`]({r['url']}) | {r['desc'] or '—'} |")
    rest = [r for r in rows if r["name"] not in CONSUMED]
    out += ["", f"## Archived ({len(rest)})", "", "| Repo | Language | Description |", "|---|---|---|"]
    for r in rest:
        out.append(f"| [`{r['name']}`]({r['url']}) | {r['lang'] or '—'} | {r['desc'] or '—'} |")
    out += ["", f"_{len(rows)} repositories, {len(CONSUMED)} consumed, {len(rest)} archived._", ""]
    print("\n".join(out))


if __name__ == "__main__":
    main()
