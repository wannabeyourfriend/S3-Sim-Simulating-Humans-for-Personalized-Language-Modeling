"""Benchmark OpenAI models on the locally-generated QA-format JSONL data. """

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_MCQ_LETTER_PATTERNS = [
    r"\$\$\s*\\boxed\{([A-Z])\}\s*\$\$",
    r"\$\\boxed\{([A-Z])\}\$",
    r"\\boxed\{([A-Z])\}",
    r"\*{0,2}\s*Final Answer\s*:?\s*\*{0,2}[\s*]*\[?([A-Z])\]?\*{0,2}",
    r"\*{0,2}\s*final answer\s*:?\s*\*{0,2}[\s*]*\[?([A-Z])\]?\*{0,2}",
    r"Answer:\s*\*{0,2}\[?([A-Z])\]?\*{0,2}",
    r"answer:\s*\*{0,2}\[?([A-Z])\]?\*{0,2}",
    r"correct answer is\s*\*{0,2}\[?([A-Z])\]?\*{0,2}",
    r"the answer is\s*\$?\\boxed\{([A-Z])\}\$?",
    r"the answer is\s*\*{0,2}\(?\[?([A-Z])\]?\)?\*{0,2}",
    r"final answer is\s*\$?\\boxed\{([A-Z])\}\$?",
    r"final answer is\s*\*{0,2}\(?\[?([A-Z])\]?\)?\*{0,2}",
    r"option\s+\*{0,2}\(?([A-Z])\)?\*{0,2}",
    r"\b([A-Z])\.\s*$",
]


def extract_mcq_letter(response: str) -> str | None:
    """Mirror p13n-eval-harness/personamem/inference.py letter extractor."""
    if not response:
        return None

    text = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)
    text = re.sub(r"</think>\s*", "", text)
    for pat in _MCQ_LETTER_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def score_bigtom(response: str, correct_letter: str, correct_text: str) -> bool:
    """Mirror p13n-eval-harness/bigtom/run.py:_score(). Substring match.

    Returns True if `a)` or `b)` (whichever is correct) appears, falling back
    to a content-match against the first 20 chars of the correct option.
    """
    if not response or not correct_letter:
        return False
    p = response.lower()
    target = f"{correct_letter.lower()})"
    other = "b)" if correct_letter.lower() == "a" else "a)"
    if target in p:
        return True
    if other in p:
        return False
    return correct_text.lower()[:20] in p


_PREFEVAL_JUDGE_PROMPT = """You are evaluating whether an AI assistant response respects a stated user preference.

<user_preference>
{preference}
</user_preference>

<user_question>
{question}
</user_question>

<assistant_response>
{response}
</assistant_response>

Score the response on three binary dimensions:
1. acknowledges: does the response explicitly reference the preference (e.g., "since you prefer..." or quoting it)? Implicit alignment without a verbal nod counts as FALSE here.
2. violates_preference: does the response make a recommendation that actively contradicts the preference? (Adding helpful tangential information that the user did not ask for is NOT a violation by itself.)
3. helpful: does the response answer the question with useful, concrete content (not platitudes, not refusal)?

Return JSON only:
{{"acknowledges": true|false, "violates_preference": true|false, "helpful": true|false, "reason": "<one sentence>"}}
"""


async def judge_prefeval(response: str, preference: str, question: str, llm) -> tuple[bool, dict]:
    """Returns (passed, raw_judge_dict). Passed = helpful AND NOT violates.

    Acknowledgement is recorded for diagnostic purposes but no longer gates
    pass/fail — implicit respect for a preference is still respect, and the
    explicit-acknowledgement requirement was systematically punishing models
    that gave content-rich answers without verbalizing the preference.
    """
    prompt = _PREFEVAL_JUDGE_PROMPT.format(
        preference=preference,
        question=question,
        response=response,
    )
    try:
        data = await llm.chat_json(
            [{"role": "system", "content": prompt}, {"role": "user", "content": "Score now."}],
            temperature=0.0,
            max_tokens=300,
            call_type="eval_prefeval_judge",
        )
    except Exception as e:
        return False, {"error": str(e)}
    if not isinstance(data, dict):
        return False, {"raw": str(data)[:200]}
    helpful = bool(data.get("helpful"))
    violates = bool(data.get("violates_preference"))
    return (helpful and not violates), data


def score_lamp(response: str, target: str) -> bool:
    """Case-insensitive exact match, with token-overlap fallback.

    The token split treats both whitespace and underscores as separators so
    snake_case labels (e.g. `financial_management`) match against the
    semantic words inside, not as opaque single tokens. Threshold is 50% to
    accept paraphrases like `Astronomy Research` ↔ `Dark Matter Research`.
    """
    if not response or not target:
        return False
    r = response.strip().lower()
    t = target.strip().lower()
    if t == r:
        return True

    if t in r:
        return True

    t_tokens = [w.strip(".,!?;:\"'") for w in re.split(r"[\s_]+", t) if len(w) >= 3]
    r_tokens = set(re.split(r"[\s_]+", r))
    if not t_tokens:
        return False
    hits = sum(1 for w in t_tokens if w in r_tokens or w in r)
    return hits / len(t_tokens) >= 0.5


async def evaluate_one(item: dict, model_llm, judge_llm, max_tokens: int) -> dict:
    """Run one QA item through the model under test and score the response."""
    msgs = item["messages"]
    meta = item["metadata"]
    style = meta["qa_style"]

    prompt_msgs = msgs[:-1]

    try:
        response = await model_llm.chat(
            prompt_msgs,
            temperature=0.0,
            max_tokens=max_tokens,
            call_type=f"eval_{style}",
        )
    except Exception as e:
        return {
            "persona_id": meta.get("persona_id"),
            "scenario_id": meta.get("scenario_id"),
            "qa_style": style,
            "error": str(e),
            "correct": False,
        }

    out = {
        "persona_id": meta.get("persona_id"),
        "scenario_id": meta.get("scenario_id"),
        "qa_style": style,
        "response": response,
    }

    if style == "personamem_mcq":
        gold = meta.get("correct_letter")
        pred = extract_mcq_letter(response)
        out["predicted_letter"] = pred
        out["gold_letter"] = gold
        out["correct"] = pred is not None and pred.upper() == (gold or "").upper()
    elif style == "bigtom_tom":
        gold_letter = meta.get("correct_letter", "").lower()

        gold_asst = msgs[-1]["content"]
        m = re.search(r"Answer:[ab]\)(.*)$", gold_asst, re.IGNORECASE | re.DOTALL)
        gold_text = m.group(1).strip() if m else ""
        out["gold_letter"] = gold_letter
        out["correct"] = score_bigtom(response, gold_letter, gold_text)
    elif style == "prefeval_gen":
        preference = meta.get("preference", "")

        question = msgs[-2]["content"] if msgs[-2]["role"] == "user" else ""
        passed, judge = await judge_prefeval(response, preference, question, judge_llm)
        out["judge"] = judge
        out["correct"] = passed
    elif style == "lamp_cls":
        gold = meta.get("gold_target") or msgs[-1]["content"]
        out["gold"] = gold
        out["correct"] = score_lamp(response, gold)
    else:
        out["correct"] = False
        out["error"] = f"unknown qa_style: {style}"
    return out


def _resolve_max_tokens(model_name: str, default: int, overrides: dict[str, int]) -> int:
    """Pick max_tokens for `model_name`. Override > family rule > default.

    Family rule: GPT-5 reasoning models silently consume tokens on hidden
    reasoning, so they need a much larger budget than chat models. We bump to
    16k by default so visible output isn't truncated to empty.
    """
    if model_name in overrides:
        return overrides[model_name]
    if "gpt-5" in model_name.lower():
        return max(default, 16384)
    return default


async def evaluate_style_for_model(
    model_name: str,
    items: list[dict],
    output_path: Path,
    judge_llm,
    concurrency: int,
    max_tokens: int,
) -> dict:
    """Run one (model, style) combination → write predictions JSONL and return summary."""
    from user_simulator.data import LLM

    model_llm = LLM(model=model_name, max_concurrent=concurrency, retries=2)
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = [None] * len(items)

    async def bounded(i: int):
        async with sem:
            try:
                results[i] = await evaluate_one(items[i], model_llm, judge_llm, max_tokens)
            except Exception as e:
                results[i] = {
                    "persona_id": items[i]["metadata"].get("persona_id"),
                    "scenario_id": items[i]["metadata"].get("scenario_id"),
                    "qa_style": items[i]["metadata"].get("qa_style"),
                    "error": f"{type(e).__name__}: {e}",
                    "correct": False,
                }

    t0 = time.time()
    await asyncio.gather(*[bounded(i) for i in range(len(items))])
    elapsed = time.time() - t0

    n = len(results)
    n_correct = sum(1 for r in results if r and r.get("correct"))
    n_errored = sum(1 for r in results if r and "error" in r)
    n_unparseable = sum(
        1
        for r in results
        if r and r.get("qa_style") == "personamem_mcq" and r.get("predicted_letter") is None
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "model": model_name,
        "qa_style": items[0]["metadata"]["qa_style"] if items else "?",
        "n": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n else 0.0,
        "n_errored": n_errored,
        "n_unparseable_mcq": n_unparseable,
        "elapsed_s": round(elapsed, 1),
        "max_tokens": max_tokens,
        "llm_calls": model_llm.calls,
        "tokens": model_llm.tokens,
    }
    return summary


def _load_jsonl_items(path: Path, sample: int | None) -> list[dict]:
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if sample is not None and sample > 0:
        items = items[:sample]
    return items


async def main(args):
    from user_simulator.data import LLM

    qa_dir = Path(args.qa_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    style_files = {p.stem: p for p in sorted(qa_dir.glob("*.jsonl")) if p.stem in args.styles}
    missing = [s for s in args.styles if s not in style_files]
    if missing:
        logger.warning("No JSONL found for styles: %s", missing)

    judge_llm = LLM(model=args.judge_model, max_concurrent=args.concurrency, retries=2)

    overrides: dict[str, int] = {}
    for spec in args.model_max_tokens or []:
        if "=" not in spec:
            raise SystemExit(f"--model-max-tokens entry {spec!r} must be model=N")
        m, v = spec.split("=", 1)
        overrides[m.strip()] = int(v.strip())

    summaries: list[dict] = []
    for style in args.styles:
        if style not in style_files:
            continue
        items = _load_jsonl_items(style_files[style], args.sample)
        logger.info("Style %s: %d items loaded from %s", style, len(items), style_files[style])
        for model_name in args.models:
            safe = model_name.replace("/", "_")
            preds_path = out_dir / f"{style}__{safe}.predictions.jsonl"
            mt = _resolve_max_tokens(model_name, args.max_tokens, overrides)
            logger.info(
                "→ evaluating %s on %s (%d items, max_tokens=%d)", model_name, style, len(items), mt
            )
            summary = await evaluate_style_for_model(
                model_name=model_name,
                items=items,
                output_path=preds_path,
                judge_llm=judge_llm,
                concurrency=args.concurrency,
                max_tokens=mt,
            )
            logger.info(
                "   %s/%s: acc=%.3f (%d/%d) errs=%d unparseable_mcq=%d in %.1fs",
                model_name,
                style,
                summary["accuracy"],
                summary["n_correct"],
                summary["n"],
                summary["n_errored"],
                summary["n_unparseable_mcq"],
                summary["elapsed_s"],
            )
            summaries.append(summary)

    summary_path = out_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    md = [
        "# QA-eval summary",
        "",
        "| model | style | n | accuracy | errs | unparseable_mcq | max_tokens | wall |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        md.append(
            f"| {s['model']} | {s['qa_style']} | {s['n']} | "
            f"{s['accuracy']:.3f} | {s['n_errored']} | {s['n_unparseable_mcq']} | "
            f"{s.get('max_tokens', '?')} | {s['elapsed_s']}s |"
        )
    (out_dir / "eval_summary.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("Summary → %s and %s", summary_path, summary_path.with_suffix(".md"))

    print("\n" + "\n".join(md))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark OpenAI models on QA-format JSONL data")
    parser.add_argument(
        "--qa-dir",
        default="output/qa/v1_demo_full",
        help="Directory containing per-style JSONL files",
    )
    parser.add_argument(
        "--output-dir", default="output/eval/v1_demo_subset", help="Where predictions + summary go"
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["personamem_mcq", "prefeval_gen", "bigtom_tom", "lamp_cls"],
        choices=["personamem_mcq", "prefeval_gen", "bigtom_tom", "lamp_cls"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-5-mini"],
        help="Models to evaluate (must be reachable via OPENAI_BASE_URL)",
    )
    parser.add_argument(
        "--judge-model", default="gpt-4.1-mini", help="Judge model for PrefEval scoring"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Items per style. Use 0 (or any non-positive) "
        "for the full set. Default 10 = subset mode.",
    )
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Default per-call max_tokens. GPT-5 family is "
        "auto-bumped to 16384 unless overridden by "
        "--model-max-tokens.",
    )
    parser.add_argument(
        "--model-max-tokens",
        nargs="+",
        default=None,
        metavar="MODEL=N",
        help="Per-model max_tokens overrides, e.g. 'gpt-5-mini=16384 gpt-4o=2048'.",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
