"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import os  # noqa: F401
import random  # noqa: F401
import re  # noqa: F401
import time  # noqa: F401
import urllib.parse  # noqa: F401
import xml.etree.ElementTree  # noqa: F401  (arXiv answers with Atom XML)

import httpx  # noqa: F401
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

ARXIV_MIN_INTERVAL = 3.0     # arXiv API etiquette: at least 3s between two calls
SUMMARY_CHARS = 600           # keep the records small so the context does not explode
FETCH_CHARS = 12000           # web_fetch truncation
RETRY_STATUS = {429, 500, 502, 503, 504}
TIMEOUT = 30.0

_last_arxiv_call = 0.0        # monotonic timestamp of the last arXiv request (same process = same host)


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def _clamp(value, low, high):
    return max(low, min(high, int(value)))


def _clean(text):
    """Collapse the newlines and the runs of spaces that XML/text sources are full of."""
    return " ".join(str(text or "").split())


def _error(exc, secret=""):
    """Every tool ends here: no exception ever reaches the agent."""
    message = f"{type(exc).__name__}: {exc}"
    if secret:                                    # never leak the Exa key through an httpx error message
        message = message.replace(secret, "***")
    return f"ERROR: {message}"


def _response_error(response):
    """Return a RetryableError for the statuses worth retrying, else None."""
    if response.status_code not in RETRY_STATUS:
        return None
    retry_after = None
    header = response.headers.get("Retry-After")
    if header:
        try:
            retry_after = float(header.strip())
        except (TypeError, ValueError):
            retry_after = None
    return RetryableError(f"HTTP {response.status_code}", retry_after)


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise                     # last attempt: no extra sleep, the error goes to the caller
            if exc.retry_after is not None:
                delay = min(float(exc.retry_after), cap)   # the server told us how long to wait
            else:
                delay = min(base * (2 ** attempt), cap) + random.uniform(0, 1)   # jitter avoids a thundering herd
            time.sleep(delay)
    raise RetryableError("unreachable")      # pragma: no cover


def _get_json(url, params, *, attempts=5, cap=30.0):
    """GET an HTTP JSON API, turning retryable transport/status errors into RetryableError."""
    def call():
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
            response = client.get(url, params=params, headers={"Accept": "application/json"})
        error = _response_error(response)
        if error is not None:
            raise error
        response.raise_for_status()
        return response.json()

    return with_retry(call, attempts=attempts, cap=cap)


def _wait_for_arxiv():
    """arXiv asks for at least 3 seconds between two API calls from the same client."""
    global _last_arxiv_call
    now = time.monotonic()
    wait = _last_arxiv_call + ARXIV_MIN_INTERVAL - now
    if wait > 0:
        time.sleep(wait)
    _last_arxiv_call = time.monotonic()


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    # the query comes from an LLM: keep only word characters so quotes/colons/AND do not break it
    terms = re.findall(r"\w+", query or "", flags=re.UNICODE)
    if not terms:
        return "NO RESULTS"
    limit = _clamp(max_results, 1, 30)

    def call():
        _wait_for_arxiv()
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
            response = client.get(ARXIV_URL, params={
                "search_query": " AND ".join(f"all:{term}" for term in terms),
                "sortBy": "submittedDate", "sortOrder": "descending", "max_results": limit,
            })
        error = _response_error(response)
        if error is not None:
            raise error
        response.raise_for_status()
        return response.text

    try:
        # a whole classroom shares one public IP: arXiv answers 429 often, so be patient here
        xml_text = with_retry(call, attempts=6, base=2.0, cap=60.0)
        root = xml.etree.ElementTree.fromstring(xml_text)
    except Exception as exc:  # noqa: BLE001
        return _error(exc)

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    records = []
    try:
        for entry in root.findall("atom:entry", ns):
            raw_id = _clean((entry.findtext("atom:id", default="", namespaces=ns)).split("/abs/")[-1])
            if not raw_id:
                continue
            paper_id = raw_id.split("v")[0]          # 2501.00001v1 -> 2501.00001
            records.append({
                "id": paper_id,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "published": _clean(entry.findtext("atom:published", default="", namespaces=ns))[:10],
                "title": _clean(entry.findtext("atom:title", default="", namespaces=ns)),
                "summary": _clean(entry.findtext("atom:summary", default="", namespaces=ns))[:SUMMARY_CHARS],
            })
    except Exception as exc:  # noqa: BLE001
        return _error(exc)

    if not records:
        return "NO RESULTS"
    return json.dumps(records, ensure_ascii=False)


def _hf_records(items, prefer_ai_summary=False):
    """Map the HF Daily/Search item shape to the record shape both tools return."""
    records = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        paper = item.get("paper") if isinstance(item.get("paper"), dict) else item
        paper_id = paper.get("id") or item.get("id")
        if not paper_id:
            continue
        summary = ""
        if prefer_ai_summary:
            summary = paper.get("ai_summary") or item.get("ai_summary") or ""
        summary = summary or paper.get("summary") or item.get("summary") or ""
        records.append({
            "id": str(paper_id),
            "url": f"https://huggingface.co/papers/{paper_id}",
            "published": _clean(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
            "title": _clean(paper.get("title") or item.get("title") or ""),
            "summary": _clean(summary)[:SUMMARY_CHARS],
            "upvotes": paper.get("upvotes") or item.get("upvotes") or 0,
            "github": paper.get("githubRepo") or item.get("githubRepo") or "",
            "stars": paper.get("githubStars") or item.get("githubStars") or 0,
        })
    return records


def _dump(records):
    return "NO RESULTS" if not records else json.dumps(records, ensure_ascii=False)


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    params = {"limit": _clamp(limit, 1, 100)}
    if date:
        params["date"] = date
    try:
        data = _get_json(HF_DAILY_URL, params)
    except Exception as exc:  # noqa: BLE001
        return _error(exc)
    records = _hf_records(data)
    needle = (keyword or "").strip().lower()
    if needle:
        records = [r for r in records if needle in f"{r['title']} {r['summary']}".lower()]
    records.sort(key=lambda r: r["upvotes"] or 0, reverse=True)
    return _dump(records)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    if not (query or "").strip():
        return "NO RESULTS"
    try:
        data = _get_json(HF_SEARCH_URL, {"q": query.strip(), "limit": _clamp(limit, 1, 50)})
    except Exception as exc:  # noqa: BLE001
        return _error(exc)
    return _dump(_hf_records(data, prefer_ai_summary=True))


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _exa_url():
    """The MCP endpoint; with a key it becomes a query parameter (so the key can leak: always redact)."""
    key = (os.getenv("EXA_API_KEY") or "").strip()
    if not key:
        return EXA_URL
    return f"{EXA_URL}?{urllib.parse.urlencode({'exaApiKey': key})}"


def _mcp_call(name, arguments, secret=""):
    """One MCP tool call: a JSON-RPC request over plain HTTP POST. Raises on error (retryable or not)."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": name, "arguments": arguments}}
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    with httpx.Client(timeout=TIMEOUT * 3, follow_redirects=True) as client:
        response = client.post(_exa_url(), json=payload, headers=headers)
    error = _response_error(response)
    if error is not None:
        raise error
    response.raise_for_status()
    body = response.text

    data = None
    for line in body.splitlines():                     # the answer is server-sent events: take the data: lines
        if line.startswith("data:"):
            try:
                data = json.loads(line[5:].strip())
            except ValueError:
                continue
    if data is None:
        try:
            data = json.loads(body)
        except ValueError:
            data = None
    if not isinstance(data, dict):
        raise RetryableError(f"unreadable MCP answer: {body[:200]}")

    if data.get("error"):
        message = json.dumps(data["error"], ensure_ascii=False)
        if secret:
            message = message.replace(secret, "***")
        raise RetryableError(f"MCP error: {message[:300]}")   # a bad call is worth one more try

    result = data.get("result") or {}
    text = "\n".join(part.get("text", "") for part in result.get("content", [])
                     if isinstance(part, dict) and part.get("type") == "text")

    # WATCH OUT: on the free tier Exa answers HTTP 200 and puts a rate-limit message in the text, flagged
    # in result._meta. Without this check the agent would take that message for page content.
    meta = result.get("_meta") or {}
    flags = json.dumps(meta).lower() if meta else ""
    rate_limited = any(token in flags for token in ("rate", "limit", "quota", "429")) or bool(
        meta and meta.get("status") in ("error", "rate_limited"))
    if not rate_limited:
        low = text.lower()
        rate_limited = ("rate limit" in low or "rate-limit" in low or "too many requests" in low
                        or "quota exceeded" in low)
    if rate_limited:
        raise RetryableError("Exa rate limited (see _meta)")
    return text


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    if not (query or "").strip():
        return "NO RESULTS"
    key = (os.getenv("EXA_API_KEY") or "").strip()
    text = None
    try:
        text = with_retry(
            lambda: _mcp_call("web_search_exa", {"query": query.strip(),
                                                 "objective": objective.strip() or f"Find pages about: {query.strip()}",
                                                 "numResults": _clamp(num_results, 1, 10)}, secret=key),
            attempts=6, base=2.0, cap=60.0)
    except Exception as exc:  # noqa: BLE001
        return _error(exc, key)
    return text.strip() or "NO RESULTS"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    if not (url or "").strip():
        return "NO RESULTS"
    key = (os.getenv("EXA_API_KEY") or "").strip()
    try:
        text = with_retry(lambda: _mcp_call("web_fetch_exa", {"urls": [url.strip()]}, secret=key),
                          attempts=6, base=2.0, cap=60.0)
    except Exception as exc:  # noqa: BLE001
        return _error(exc, key)
    return text.strip()[:FETCH_CHARS] or "NO RESULTS"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")