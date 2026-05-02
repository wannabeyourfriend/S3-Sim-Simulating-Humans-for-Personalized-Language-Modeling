"""Rewrite v1 QA items into harder, persona-grounded v2 items using gpt-4o.

For each input SFT line (from `output/.../qa/{style}.jsonl`), reverse-engineer
the underlying question/options/preference, prompt gpt-4o with the per-style
rewrite YAML, then re-render the v2 SFT line preserving the gold label and
the eval-harness-expected message shape.

Outputs land in `--output-dir/{style}.jsonl`. The original v1 line is kept
in `metadata.v1_*` for diffing.

Usage:
    set -a && source .env.openai && set +a && \\
      uv run python run_qa_rewrite.py \\
        --qa-dir output/release_us_1k/qa \\
        --output-dir output/release_us_1k/qa_v2 \\
        --rewriter-model gpt-4o \\
        --concurrency 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PERSONAMEM_RECALL_SUFFIX = (
    " Please recall my related preferences from our conversation history "
    "to give personalized responses."
)
_BASE_SFT_INSTRUCTION = (
    "You are a personalized AI assistant. Use the conversation context to "
    "give a response that reflects the user's preferences."
)
_PERSONAMEM_OPTIONS_INTRO = "Please choose the best answer from the following options:"
_PERSONAMEM_OPTIONS_OUTRO = (
    "Think step by step about which answer best fits the user's "
    "query and conversation context. Provide your reasoning first, "
    "then give your final answer as 'Final Answer: [Letter]'"
)


def _personamem_options_block(options: list[str]) -> str:
    parts = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
    return (
        f"{_PERSONAMEM_OPTIONS_INTRO}\n\n" + "\n".join(parts) + f"\n\n{_PERSONAMEM_OPTIONS_OUTRO}"
    )


_OPTION_RE = re.compile(r"^([A-D])\.\s+(.*)$", re.MULTILINE)


def _parse_personamem(item: dict, ctx: dict | None = None) -> dict | None:
    """Return {profile_block, conversation_excerpt, user_query, options[4],
    correct_letter, correct_text}. `ctx` is unused (this style carries its
    own profile + conversation in the SFT messages)."""
    msgs = item["messages"]
    correct_letter = item["metadata"].get("correct_letter")
    if not correct_letter:
        return None
    sys_msg = msgs[0]["content"]
    m = re.search(r"<persona>\s*\n(.*?)\n</persona>", sys_msg, re.DOTALL)
    profile_block = m.group(1).strip() if m else ""
    history = msgs[1:-2]
    conv_excerpt = "\n".join(
        f"{('User' if m['role'] == 'user' else 'Assistant')}: {m['content']}" for m in history
    )
    last_user = msgs[-2]["content"]
    if _PERSONAMEM_RECALL_SUFFIX in last_user:
        head, _, tail = last_user.partition(_PERSONAMEM_RECALL_SUFFIX)
        user_query = head.strip()
        opts_part = tail
    else:
        m = re.search(re.escape(_PERSONAMEM_OPTIONS_INTRO), last_user)
        if not m:
            return None
        user_query = last_user[: m.start()].strip()
        opts_part = last_user[m.start() :]
    options: list[str] = ["", "", "", ""]
    for letter, text in _OPTION_RE.findall(opts_part):
        idx = ord(letter) - ord("A")
        if 0 <= idx < 4:
            options[idx] = text.strip().split("\n\n")[0].strip()
    if not all(options):
        return None
    correct_idx = ord(correct_letter) - ord("A")
    return {
        "profile_block": profile_block,
        "conversation_excerpt": conv_excerpt,
        "user_query": user_query,
        "options": options,
        "correct_letter": correct_letter,
        "correct_text": options[correct_idx],
    }


def _parse_prefeval(item: dict, ctx: dict | None = None) -> dict | None:
    msgs = item["messages"]
    if len(msgs) < 5:
        return None
    ctx = ctx or {}
    return {
        "profile_block": ctx.get("profile_block", ""),
        "conversation_excerpt": ctx.get("conversation_excerpt", ""),
        "preference": msgs[1]["content"],
        "question": msgs[3]["content"],
        "response": msgs[4]["content"],
    }


def _parse_bigtom(item: dict, ctx: dict | None = None) -> dict | None:
    msgs = item["messages"]
    correct_letter = item["metadata"].get("correct_letter", "").lower()
    if correct_letter not in {"a", "b"}:
        return None
    user_msg = msgs[1]["content"]
    ctx = ctx or {}

    m = re.search(r"\n\nChoose one of the following:\n", user_msg)
    if not m:
        return None
    pre = user_msg[: m.start()]
    post = user_msg[m.end() :]
    parts = pre.split("\n\n", 1)
    if len(parts) != 2:
        return None
    narrative, question = parts[0].strip(), parts[1].strip()
    opt_a, opt_b = "", ""
    for line in post.splitlines():
        line = line.strip()
        if line.lower().startswith("a)"):
            opt_a = line[2:].strip()
        elif line.lower().startswith("b)"):
            opt_b = line[2:].strip()
    if not opt_a or not opt_b:
        return None
    return {
        "profile_block": ctx.get("profile_block", ""),
        "conversation_excerpt": ctx.get("conversation_excerpt", ""),
        "narrative": narrative,
        "question": question,
        "option_a": opt_a,
        "option_b": opt_b,
        "correct_letter": correct_letter,
    }


_LAMP_ITEM_RE = re.compile(r"^- input:\s*(.*?)\s*\|\s*output:\s*(.*)$", re.MULTILINE)


def _parse_lamp(item: dict, ctx: dict | None = None) -> dict | None:
    ctx = ctx or {}
    msgs = item["messages"]
    user_msg = msgs[1]["content"]
    items = [{"input": i.strip(), "output": o.strip()} for i, o in _LAMP_ITEM_RE.findall(user_msg)]
    if len(items) < 2:
        return None
    m = re.search(r"New input:\s*(.*?)\nProduce the output\.", user_msg, re.DOTALL)
    if not m:
        return None
    query = m.group(1).strip()
    gold = msgs[-1]["content"].strip()
    family = item["metadata"].get("task_family", "tag_classify")
    profile_items_str = "\n".join(
        f"- input: {it['input']} | output: {it['output']}" for it in items
    )
    return {
        "profile_block": ctx.get("profile_block", ""),
        "profile_items": items,
        "profile_items_str": profile_items_str,
        "query": query,
        "gold_target": gold,
        "task_family": family,
    }


def _render_personamem_v2(orig_item: dict, parsed: dict, rewritten: dict) -> dict | None:
    options_dict = rewritten.get("options") or {}
    if not isinstance(options_dict, dict):
        return None
    options = [options_dict.get(L, "").strip() for L in "ABCD"]
    if not all(options):
        return None
    new_query = str(rewritten.get("user_query", "")).strip()
    answer = str(rewritten.get("reasoning_trace_and_answer", "")).strip()
    correct_letter = parsed["correct_letter"]
    if not new_query or not answer:
        return None
    if f"Final Answer: {correct_letter}" not in answer:
        answer = f"{answer}\n\nFinal Answer: {correct_letter}"

    sys_msg = orig_item["messages"][0]["content"]
    history = orig_item["messages"][1:-2]
    user_msg = (
        new_query.rstrip() + _PERSONAMEM_RECALL_SUFFIX + "\n\n" + _personamem_options_block(options)
    )
    new_messages = [{"role": "system", "content": sys_msg}]
    new_messages.extend(history)
    new_messages.append({"role": "user", "content": user_msg})
    new_messages.append({"role": "assistant", "content": answer})
    md = dict(orig_item["metadata"])
    md["v1_user_query"] = parsed["user_query"]
    md["v1_options"] = parsed["options"]
    md["rewrite_model"] = "gpt-4o"
    return {"messages": new_messages, "metadata": md}


def _render_prefeval_v2(orig_item: dict, parsed: dict, rewritten: dict) -> dict | None:
    new_pref = str(rewritten.get("preference", "")).strip()
    new_q = str(rewritten.get("question", "")).strip()
    new_resp = str(rewritten.get("assistant_response", "")).strip()
    ack_quote = str(rewritten.get("acknowledge_quote", "")).strip()
    if not new_pref or not new_q or not new_resp:
        return None

    resp_lower = new_resp.lower()
    if not (ack_quote and ack_quote.lower() in resp_lower):
        pref_tokens = {t.lower().strip(".,!?;:\"'") for t in new_pref.split() if len(t) >= 4}
        hits = sum(1 for t in pref_tokens if t and t in resp_lower)
        if hits < 2:
            return None
    new_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": new_pref},
        {"role": "assistant", "content": "Got it — I'll keep that in mind for our conversation."},
        {"role": "user", "content": new_q},
        {"role": "assistant", "content": new_resp},
    ]
    md = dict(orig_item["metadata"])
    md["preference"] = new_pref
    md["v1_preference"] = parsed["preference"]
    md["v1_question"] = parsed["question"]
    md["rewrite_model"] = "gpt-4o"
    return {"messages": new_messages, "metadata": md}


def _render_bigtom_v2(orig_item: dict, parsed: dict, rewritten: dict) -> dict | None:
    narrative = str(rewritten.get("narrative", "")).strip()
    question = str(rewritten.get("question", "")).strip()
    opt_a = str(rewritten.get("option_a", "")).strip()
    opt_b = str(rewritten.get("option_b", "")).strip()
    answer = str(rewritten.get("reasoning_trace_and_answer", "")).strip()
    correct_letter = parsed["correct_letter"]
    correct_text = opt_a if correct_letter == "a" else opt_b
    if not all([narrative, question, opt_a, opt_b, answer]):
        return None
    expected_tail = f"Answer:{correct_letter}){correct_text}"
    if expected_tail not in answer:
        answer = f"{answer}\n\n{expected_tail}"
    user_text = f"{narrative}\n\n{question}\n\nChoose one of the following:\na) {opt_a}\nb) {opt_b}"
    sys_msg = orig_item["messages"][0]["content"]
    new_messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer},
    ]
    md = dict(orig_item["metadata"])
    md["v1_narrative"] = parsed["narrative"]
    md["v1_question"] = parsed["question"]
    md["rewrite_model"] = "gpt-4o"
    return {"messages": new_messages, "metadata": md}


def _render_lamp_v2(orig_item: dict, parsed: dict, rewritten: dict) -> dict | None:
    items = rewritten.get("profile_items") or []
    if not isinstance(items, list) or len(items) < 2:
        return None
    item_lines = []
    for it in items:
        if not isinstance(it, dict):
            continue
        ip = str(it.get("input", "")).strip()
        op = str(it.get("output", "")).strip()
        if ip and op:
            item_lines.append(f"- input: {ip} | output: {op}")
    if not item_lines:
        return None
    query = str(rewritten.get("query", "")).strip()
    answer = str(rewritten.get("reasoning_trace_and_answer", "")).strip()
    gold = parsed["gold_target"]
    if not query or not answer:
        return None

    answer = answer.rstrip()
    last_line = answer.splitlines()[-1].strip() if answer else ""

    if not last_line.lower().endswith(gold.lower()):
        answer = answer + f"\n{gold}"
    user_text = (
        "Here are some past items belonging to this user:\n"
        + "\n".join(item_lines)
        + f"\n\nNew input: {query}\nProduce the output."
    )
    sys_msg = orig_item["messages"][0]["content"]
    new_messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer},
    ]
    md = dict(orig_item["metadata"])
    md["v1_query"] = parsed["query"]
    md["gold_target"] = gold
    md["rewrite_model"] = "gpt-4o"
    return {"messages": new_messages, "metadata": md}


_PARSERS = {
    "personamem_mcq": _parse_personamem,
    "prefeval_gen": _parse_prefeval,
    "bigtom_tom": _parse_bigtom,
    "lamp_cls": _parse_lamp,
}

_RENDERERS = {
    "personamem_mcq": _render_personamem_v2,
    "prefeval_gen": _render_prefeval_v2,
    "bigtom_tom": _render_bigtom_v2,
    "lamp_cls": _render_lamp_v2,
}

_PROMPTS = {
    "personamem_mcq": "qa_rewrite_personamem",
    "prefeval_gen": "qa_rewrite_prefeval",
    "bigtom_tom": "qa_rewrite_bigtom",
    "lamp_cls": "qa_rewrite_lamp",
}


def _render_prompt(style: str, parsed: dict, tmpl: str) -> str:
    from user_simulator.prompts import render

    if style == "personamem_mcq":
        opts_block = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(parsed["options"]))
        return render(
            tmpl,
            profile_block=parsed["profile_block"],
            conversation_excerpt=parsed["conversation_excerpt"],
            original_user_query=parsed["user_query"],
            original_options_block=opts_block,
            correct_letter=parsed["correct_letter"],
            original_correct_text=parsed["correct_text"],
        )
    if style == "prefeval_gen":
        return render(
            tmpl,
            profile_block=parsed["profile_block"],
            conversation_excerpt=parsed["conversation_excerpt"],
            original_preference=parsed["preference"],
            original_question=parsed["question"],
            original_response=parsed["response"],
        )
    if style == "bigtom_tom":
        return render(
            tmpl,
            profile_block=parsed["profile_block"],
            conversation_excerpt=parsed["conversation_excerpt"],
            original_narrative=parsed["narrative"],
            original_question=parsed["question"],
            original_option_a=parsed["option_a"],
            original_option_b=parsed["option_b"],
            correct_letter=parsed["correct_letter"],
        )
    if style == "lamp_cls":
        return render(
            tmpl,
            profile_block=parsed["profile_block"],
            original_profile_items=parsed["profile_items_str"],
            original_query=parsed["query"],
            gold_target=parsed["gold_target"],
            task_family=parsed["task_family"],
        )
    raise ValueError(style)


async def _rewrite_one(
    style: str, orig_item: dict, tmpl: str, llm, ctx_lookup: dict | None = None
) -> dict | None:
    md = orig_item.get("metadata", {})
    key = (md.get("persona_id", ""), md.get("scenario_id", ""))
    ctx = (ctx_lookup or {}).get(key)
    parsed = _PARSERS[style](orig_item, ctx)
    if parsed is None:
        return None
    sys_prompt = _render_prompt(style, parsed, tmpl)
    try:
        rewritten = await llm.chat_json(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "Generate the rewritten JSON now."},
            ],
            temperature=0.7,
            max_tokens=2200,
            call_type=f"rewrite_{style}",
        )
    except Exception as e:
        logger.warning(
            "rewrite call failed for %s/%s/%s: %s",
            orig_item["metadata"].get("persona_id"),
            orig_item["metadata"].get("scenario_id"),
            style,
            e,
        )
        return None
    return _RENDERERS[style](orig_item, parsed, rewritten)


def _build_ctx_lookup(conv_root: Path) -> dict:
    """Map (persona_id, scenario_id) → {profile_block, conversation_excerpt}.

    Walks every *.json under conv_root that looks like an S³-Sim conversation
    (has persona_id + prompt_id + conversation). Used to inject persona +
    convo context into rewrite prompts for styles whose SFT lines don't carry it.
    """
    lookup: dict = {}
    for p in conv_root.rglob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        pid = d.get("persona_id")
        sid = d.get("prompt_id")
        if not pid or not sid:
            continue
        prof = d.get("profile_summary", "") or ""
        conv = d.get("conversation", []) or []
        excerpt = "\n".join(
            f"{('User' if m['role'] == 'user' else 'Assistant')}: {m['content']}" for m in conv[:6]
        )
        lookup[(pid, sid)] = {"profile_block": prof, "conversation_excerpt": excerpt}
    logger.info("Built context lookup for %d (persona, scenario) pairs", len(lookup))
    return lookup


async def main(args):
    from user_simulator.data import LLM
    from user_simulator.prompts import load_prompt

    qa_dir = Path(args.qa_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx_lookup: dict = {}
    if args.conversations_dir:
        ctx_lookup = _build_ctx_lookup(Path(args.conversations_dir).resolve())

    llm = LLM(model=args.rewriter_model, max_concurrent=args.concurrency, retries=2)
    logger.info("Rewriter model: %s; concurrency=%d", args.rewriter_model, args.concurrency)

    counter = {"in": 0, "out": 0, "parse_fail": 0, "rewrite_fail": 0}

    for style in args.styles:
        in_path = qa_dir / f"{style}.jsonl"
        out_path = out_dir / f"{style}.jsonl"
        if not in_path.exists():
            logger.warning("Skipping %s: no input file at %s", style, in_path)
            continue
        items = [
            json.loads(l) for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        if args.sample:
            items = items[: args.sample]
        logger.info("Style %s: %d items in", style, len(items))
        counter["in"] += len(items)

        tmpl = load_prompt(_PROMPTS[style])
        sem = asyncio.Semaphore(args.concurrency)
        results: list[dict | None] = [None] * len(items)
        progress = {"done": 0}

        async def bounded(i: int, it: dict):
            async with sem:
                v2 = await _rewrite_one(style, it, tmpl, llm, ctx_lookup)
                results[i] = v2
                progress["done"] += 1
                if progress["done"] % 50 == 0:
                    logger.info("  %s progress: %d/%d", style, progress["done"], len(items))

        t0 = time.time()
        await asyncio.gather(*[bounded(i, it) for i, it in enumerate(items)])
        elapsed = time.time() - t0

        n_written = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                if r is None:
                    continue
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_written += 1
        n_dropped = len(items) - n_written
        counter["out"] += n_written
        counter["rewrite_fail"] += n_dropped
        logger.info(
            "  %s: %d/%d written → %s (dropped=%d, %.1fs)",
            style,
            n_written,
            len(items),
            out_path,
            n_dropped,
            elapsed,
        )

    logger.info(
        "Done: %d in, %d out (%.1f%% kept), %d dropped",
        counter["in"],
        counter["out"],
        100 * counter["out"] / max(counter["in"], 1),
        counter["rewrite_fail"],
    )
    logger.info("LLM stats: %s", llm.stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite v1 QA items into harder v2 items")
    parser.add_argument("--qa-dir", required=True, help="Directory with v1 {style}.jsonl files")
    parser.add_argument("--output-dir", required=True, help="Where v2 {style}.jsonl files go")
    parser.add_argument(
        "--conversations-dir",
        default=None,
        help="Root dir of source conversation JSONs (used to inject "
        "profile_summary + conversation_excerpt into rewrite "
        "prompts for prefeval/bigtom/lamp).",
    )
    parser.add_argument("--rewriter-model", default="gpt-4o")
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["personamem_mcq", "prefeval_gen", "bigtom_tom", "lamp_cls"],
        choices=["personamem_mcq", "prefeval_gen", "bigtom_tom", "lamp_cls"],
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="Process only first N items per style (smoke)"
    )
    parser.add_argument("--concurrency", type=int, default=30)
    args = parser.parse_args()
    asyncio.run(main(args))
