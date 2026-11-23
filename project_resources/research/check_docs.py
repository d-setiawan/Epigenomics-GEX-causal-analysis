#!/usr/bin/env python3
"""Documentation drift checker — run before committing doc or path changes.

Docs rot when the same fact lives in two places and only one gets updated. This
enforces the ownership model in README.md ("Documentation ownership") mechanically:

  1. LINKS      every relative markdown link (and #anchor) resolves
  2. PATHS      no references to the pre-2026-07-20 directory layout
  3. FACTS      no known-false claims (the registry below is the thing to extend)
  4. ORPHANS    every doc is reachable from some other doc

Adding a fact to FACTS is how a correction becomes permanent: once a claim is known
false, this fails the build if anyone reintroduces it.

Usage:  python project_resources/research/check_docs.py [--quiet]
Exit:   0 clean, 1 problems found.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Directories that are outputs, vendored, or historical snapshots — never linted.
SKIP_PARTS = {
    ".git",
    ".lake",
    ".pytest_cache",
    "graphify-out",
    "__pycache__",
    "papers",
    "reports",
    "outputs",
}

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.M)

# --- 2. paths that no longer exist after the 2026-07-20 restructure ----------------
STALE_PATHS = [
    (r"\bsc_scripts/", "sc_scripts/ -> pipelines/empirical/data/scripts/"),
    (r"pipelines/empirical/analysis/evaluation/", "pipelines/empirical/analysis/evaluation/ -> pipelines/empirical/evaluation/"),
    (r"pipelines/empirical/analysis/notes/", "pipelines/empirical/analysis/notes/ -> pipelines/empirical/analysis/results/"),
    (r"pipelines/empirical/analysis/outputs/evaluation/", "-> pipelines/empirical/evaluation/outputs/"),
    (r"(?<![/\w])notes/(?!\w*\.py)", "notes/ -> project_resources/research/ (or <section>/results/)"),
]

# Stale-path rules that must NOT fire for project_resources/research living inside project_resources/research/, where `../X.md`
# is the correct way to reach a sibling of the project_resources/research root.
STALE_PATHS_OUTSIDE_DOCS = [
    (r"\]\(\.\./MISSION\.md\)", "MISSION.md now lives in project_resources/research/"),
    (r"\]\(\.\./ROADMAP\.md\)", "ROADMAP.md now lives in project_resources/research/"),
]

# --- 3. claims known to be false. Extend this whenever a fact is corrected. --------
# (regex, why it is wrong, files allowed to mention it — i.e. where it is being refuted)
FACTS = [
    (r"donor-level block[- ]bootstrap(?!.{0,120}(unavailable|impossible|not available|amended|closed))",
     "Donor blocks are impossible — no shared donor structure across modalities (LIMITATIONS L1). "
     "Do not list this as pending work.",
     {"project_resources/research/LIMITATIONS.md", "project_resources/research/ROADMAP.md"}),
    (r"invariance across donors",
     "Donor-invariance is not achievable (LIMITATIONS L1); the invariance axis is cross-cell-type (ICP).",
     {"project_resources/research/LIMITATIONS.md", "project_resources/research/ROADMAP.md", "project_resources/research/MISSION.md"}),
    (r"GFCI[^.\n]{0,60}not yet wired",
     "GFCI was wired 2026-07-09 and its Knowledge object 2026-07-15.",
     set()),
    (r"[Oo]nly 2 of 3[^.\n]{0,40}axes",
     "All three peak->gene axes (rE2G, PCHi-C, eQTL) are wired.",
     set()),
    (r"no target-aware peak-peak validator yet|peak-peak edges .{0,40}can'?t be validated",
     "The element<->element validator exists (pipelines/empirical/evaluation/scripts/peak_peak_validator.py).",
     set()),
    (r"screen(ing|)[^.\n]{0,60}signal[- ]share[^.\n]{0,60}(improve|stabilis|stabiliz|fix)",
     "Signal-share screening was tested and FAILED: at matched panel size it produced fewer "
     "edges than random subsets and zero partition-stable edges "
     "(pipelines/simulation/aggregation_identifiability/results/crossmark_identifiability.md S6, LIMITATIONS L15).",
     {"project_resources/research/LIMITATIONS.md", "pipelines/simulation/aggregation_identifiability/results/crossmark_identifiability.md"}),
    (r"proportionality completion[^.\n]{0,60}(licensed|justified|validated|holds)",
     "Sigma_within ~ c*Sigma_between does NOT transfer across marks: leave-one-mark-out "
     "R^2 = 0.013 (continuum) / 0.211 (discrete). The completion is refuted "
     "(pipelines/simulation/aggregation_identifiability/results/crossmark_identifiability.md S5c).",
     {"pipelines/simulation/aggregation_identifiability/results/crossmark_identifiability.md"}),
    # Match the ASSERTION only.  Requiring "are/is" immediately before the claim
    # keeps this off the true negated form ("are not on the same scale") and off
    # the hyphenated heading anchor, both of which are correct statements.
    (r"cross-mark[^.\n]{0,40}\b(are|is) (on the same scale|directly comparable)",
     "Same-mark covariance entries carry a sampling term that cross-mark entries do not "
     "(LIMITATIONS L14). They are not on the same scale and must not be compared directly.",
     {"project_resources/research/LIMITATIONS.md", "pipelines/simulation/aggregation_identifiability/results/crossmark_identifiability.md"}),
]


def slug(text: str) -> str:
    """GitHub-style heading anchor."""
    text = re.sub(r"`|\*", "", text).strip().lower()  # keep '_': GitHub keeps it in slugs
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s]+", "-", text)


def docs() -> list[Path]:
    return sorted(
        p for p in REPO.rglob("*.md")
        if not any(part in SKIP_PARTS for part in p.relative_to(REPO).parts)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    files = docs()
    anchors: dict[Path, set[str]] = {
        f: {slug(h) for h in HEADING_RE.findall(f.read_text(errors="ignore"))} for f in files
    }
    problems: list[str] = []
    linked_to: set[Path] = set()
    n_links = 0

    for f in files:
        rel = f.relative_to(REPO).as_posix()
        text = f.read_text(errors="ignore")

        # 1. links
        for target in LINK_RE.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            n_links += 1
            path_part, _, anchor = target.partition("#")
            dest = (f.parent / path_part).resolve()
            if not dest.exists():
                problems.append(f"[link]   {rel}\n           broken -> {target}")
                continue
            if dest.suffix == ".md":
                linked_to.add(dest)
            if anchor and dest.suffix == ".md" and anchor not in anchors.get(dest, set()):
                problems.append(f"[anchor] {rel}\n           no heading '#{anchor}' in {path_part}")

        # 2. stale paths
        rules = list(STALE_PATHS)
        if not rel.startswith("project_resources/research/"):
            rules += STALE_PATHS_OUTSIDE_DOCS
        for pat, fix in rules:
            for m in re.finditer(pat, text):
                line = text[: m.start()].count("\n") + 1
                problems.append(f"[path]   {rel}:{line}\n           '{m.group(0)}' — {fix}")

        # 3. known-false claims
        for pat, why, allowed in FACTS:
            if rel in allowed:
                continue
            for m in re.finditer(pat, text):
                line = text[: m.start()].count("\n") + 1
                problems.append(f"[fact]   {rel}:{line}\n           '{m.group(0)[:70]}' — {why}")

    # 4. orphans
    roots = {REPO / "README.md"}
    for f in files:
        if f in roots or f in linked_to:
            continue
        problems.append(f"[orphan] {f.relative_to(REPO)}\n           not linked from any other doc")

    if not a.quiet:
        print(f"checked {len(files)} docs, {n_links} relative links")
    if problems:
        print(f"\n{len(problems)} problem(s):\n")
        for p in problems:
            print("  " + p)
        return 1
    print("clean — no doc drift detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
