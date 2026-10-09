"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

# The whole report (standard library only, no third-party markdown parser).
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)          # fenced blocks and code spans: not citations
_LINK = re.compile(r"\[([^\]\n]*)\]\([^)\n]*\)")                 # [3](url): a markdown link, not a citation
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")         # [3]  [1, 2]  [1-3]  [2-3]; not [3](link)
_URL = re.compile(r"https?://[^\s<>)\]]+")


def _group_numbers(group):
    """'1, 2' / '1-3' -> [1, 2] / [1, 2, 3]."""
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def _split_sections(report_text):
    """Return (body, reference_lines). The body is everything before the last `## References` heading."""
    matches = list(_REF_HEADING.finditer(report_text))
    if not matches:
        return report_text, None
    body = report_text[: matches[-1].start()]
    tail = report_text[matches[-1].end():]
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    return body, lines


def _strip_non_citations(text):
    """Blank out code blocks/spans and markdown links so their [..] are not counted as citations."""
    segments = _CODE.split(text)                      # odd indexes are code
    out = []
    for i, segment in enumerate(segments):
        if i % 2:
            out.append(" " * len(segment))
        else:
            out.append(_LINK.sub(lambda m: " " * len(m.group(0)), segment))
    return "".join(out)


def _cited_numbers(text):
    """The set of numbers cited as [n] in `text` (code blocks and markdown links excluded)."""
    numbers = []
    for match in _GROUP.finditer(_strip_non_citations(text)):
        for n in _group_numbers(match.group(1)):
            if n not in numbers:
                numbers.append(n)
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]
    if not isinstance(report_text, str):
        return ["the report is not text"]

    # --- 1. shape of sources.json -------------------------------------------------
    by_n = {}
    seen_urls = {}
    for index, entry in enumerate(sources):
        if not isinstance(entry, dict):
            problems.append(f"sources[{index}] is not an object")
            continue
        n = entry.get("n")
        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"sources[{index}]: n={n!r} is not an integer")
            continue
        if n in by_n:
            problems.append(f"source number [{n}] appears twice in sources.json")
        by_n[n] = entry
        url = entry.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source [{n}]: url {url!r} does not start with http:// or https://")
        elif url in seen_urls:
            problems.append(f"source [{n}]: url {url} is already used by source [{seen_urls[url]}]")
        else:
            seen_urls[url] = n

    # --- 2. the References section ----------------------------------------------
    body, ref_lines = _split_sections(report_text)
    if ref_lines is None:
        problems.append("the report has no `## References` heading")
        ref_lines = []
    ref_by_n = {}
    for line in ref_lines:
        match = re.match(r"^\[(\d+)\]\s*(.*)$", line)
        if not match:
            continue                        # prose lines under the heading are allowed
        n = int(match.group(1))
        rest = match.group(2)
        if n in ref_by_n:
            problems.append(f"reference line [{n}] appears more than once")
        ref_by_n[n] = rest

    for n in sorted(by_n):
        if n not in ref_by_n:
            problems.append(f"source [{n}] has no reference line")
    for n in sorted(ref_by_n):
        if n not in by_n:
            problems.append(f"reference line [{n}] does not match any source in sources.json")

    for n, rest in sorted(ref_by_n.items()):
        urls = _URL.findall(rest)
        if not urls:
            problems.append(f"reference line [{n}] contains no URL")
        elif len(urls) > 1:
            problems.append(f"reference line [{n}] contains {len(urls)} URLs (exactly one is allowed)")
        elif n in by_n and urls[0].rstrip(".,);") != str(by_n[n].get("url", "")).rstrip(".,);"):
            problems.append(f"reference line [{n}] URL {urls[0]} != source url {by_n[n].get('url')}")

    # --- 3. citations in the body ------------------------------------------------
    cited = _cited_numbers(body)
    for n in cited:
        if n not in by_n:
            problems.append(f"[{n}] cited in the body but missing from sources.json")
    for n in sorted(by_n):
        if n not in cited:
            problems.append(f"source [{n}] never cited in the body")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))