"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
from collections import Counter  # noqa: F401
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator

# The model provider rate-limits too (free tiers do). The sandbox files survive, so a transient failure is retried
# in the SAME sandbox: the agent re-reads its notes instead of losing the whole run.
TRANSIENT = ("429", "rate limit", "ratelimit", "resource_exhausted", "timeout", "timed out",
             "overloaded", "502", "503", "529", "connection reset", "internal server error")
RUN_ATTEMPTS = 4


def _is_transient(exc):
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(token in text for token in TRANSIENT)


def invoke_agent(agent, topic):
    """Invoke the lead agent, retrying a transient provider failure (429/timeout/5xx) in the same sandbox."""
    delay = 30.0
    for attempt in range(1, RUN_ATTEMPTS + 1):
        try:
            return agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})   # deliberate: one run is many steps, not 9999
        except Exception as exc:  # noqa: BLE001
            if attempt == RUN_ATTEMPTS or not _is_transient(exc):
                raise
            print(f"[research] transient model error ({type(exc).__name__}), retry {attempt}/{RUN_ATTEMPTS - 1} "
                  f"in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 180.0)


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[^\w]+", "-", str(topic or "").strip().lower(), flags=re.UNICODE).strip("-_")
    slug = slug[:60].strip("-_")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Research topic: {topic}\n\n"
        f"Produce a survey report about \"{topic}\" following your instructions: plan with write_todos, delegate one "
        f"task per sub-question to the researcher subagent (parallel, self-contained delegation messages), check and "
        f"merge their notes into {SOURCES_PATH}, write the body of {REPORT_PATH} following REPORT_TEMPLATE.md "
        f"(no `## References` section), run `python3 {FINALIZER_PATH}`, then `python3 {VALIDATOR_PATH}` until it "
        "prints OK, and finally have the citation-checker spot-check a few claims."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    tool_calls = Counter()
    subagent_calls = 0
    input_tokens = 0
    output_tokens = 0
    for message in messages or []:
        calls = getattr(message, "tool_calls", None) or (message.get("tool_calls") if isinstance(message, dict) else None)
        for call in calls or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            tool_calls[name] += 1
            if name == "task":
                subagent_calls += 1
        usage = getattr(message, "usage_metadata", None)
        if usage is None and isinstance(message, dict):
            usage = message.get("usage_metadata")
        if usage:
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
    return {
        "model": model_name,
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_calls),
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    raw_report = files.get(REPORT_PATH)
    raw_sources = files.get(SOURCES_PATH)
    report_text = (raw_report or b"").decode("utf-8", errors="replace") if raw_report else ""
    if not report_text.strip():
        raise RuntimeError(f"the agent produced no report at {REPORT_PATH}")
    if not raw_sources:
        raise RuntimeError(f"the agent produced no {SOURCES_PATH}")
    try:
        sources = json.loads(raw_sources.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"{SOURCES_PATH} is not valid JSON: {exc}") from exc
    if not isinstance(sources, list) or not sources:
        raise RuntimeError(f"{SOURCES_PATH} is empty: no source to cite")

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    meta = summarize(messages, elapsed, model_name)
    meta.update({
        "topic": topic,
        "n_sources": len(sources),
        "source_families": sorted({str(entry.get("source", "")) for entry in sources if isinstance(entry, dict)}),
    })
    (reports_dir / f"{slug}.sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = reports_dir / f"{slug}.md"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic).

    PSEUDO-CODE:
      empty topic -> print usage to stderr, return 2
      model = make_model(); start = time.monotonic()
      with open_sandbox() as backend:                # the sandbox is always cleaned up, even on errors
          backend.execute("mkdir -p <WORKDIR>/research/notes <WORKDIR>/report")
          upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
          agent = build_lead_agent(backend, model)
          result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})
          save_outputs(...); on RuntimeError print "FAILED: ..." to stderr and return 1
      print where the report was saved; return 0
    """
    if not (topic or "").strip():
        print('usage: python research.py "survey about world model"', file=sys.stderr)
        return 2

    model = make_model()
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or str(model)
    start = time.monotonic()
    with open_sandbox() as backend:                    # the sandbox is always stopped/removed, even on errors
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        # only the validator and the finalizer go up: no .env, no API key ever enters the sandbox
        upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                         FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
        agent = build_lead_agent(backend, model)
        result = invoke_agent(agent, topic)
        elapsed = time.monotonic() - start
        try:
            report_path = save_outputs(backend, topic, result["messages"], elapsed, model_name)
        except RuntimeError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1

    meta = json.loads(report_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    print(f"report:  {report_path}")
    print(f"sources: {meta['n_sources']} from {meta['source_families']}")
    print(f"time:    {meta['elapsed_s']}s   lead tokens: {meta['tokens']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))