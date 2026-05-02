"""Output parsers for the user-simulator turn.

Pure functions — regex over strings. The user-simulator output format is:

    <think>...</think>
    <user_state>...</user_state>
    <message>...</message>

with optional `<|End Conversation|>` / `<|Continue Conversation|>` tags inside
the message. Parsing is fault-tolerant: the user_state extractor falls back
through three strategies (closed tags → unclosed tag → header heuristic) so a
single missing close-tag doesn't lose a turn.
"""
from __future__ import annotations

import json
import re


def _strip_tags(text: str) -> str:
    return re.sub(r"</?(?:think|user_state|message|report|state)>", "", text).strip()


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        return json.loads(m.group(1))
    s, e = text.find("{"), text.rfind("}") + 1
    if s >= 0 and e > s:
        return json.loads(text[s:e])
    raise json.JSONDecodeError("No JSON object", text, 0)


def _extract_end_signal(msg: str) -> tuple[str, bool]:
    """Strip continuation/end tags, return (clean_msg, wants_to_end)."""
    msg = msg.strip()
    for tag, is_end in [("<|End Conversation|>", True), ("<|Continue Conversation|>", False)]:
        if msg.startswith(tag):
            return msg[len(tag):].strip(), is_end
    if "\n" in msg:
        first, rest = msg.split("\n", 1)
        first = first.strip()
        if first == "<|End Conversation|>":
            return rest.strip(), True
        if first == "<|Continue Conversation|>":
            return rest.strip(), False
    return msg, False


def _parse_user_output(raw: str) -> dict:
    """Parse <user_state> and <message> from user simulator output.

    Three extraction strategies for user_state, ordered by specificity:
      1. <user_state>...</user_state>  (exact tags)
      2. <user_state>...              (unclosed tag, stops at <message> or EOF)
      3. # User State Report...       (model omitted tags entirely)
    """
    result = {"think": "", "user_state": "", "message": "", "wants_to_end": False}

    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if m:
        result["think"] = m.group(1).strip()

    m = re.search(r"<user_state>(.*?)</user_state>", raw, re.DOTALL)
    if m:
        result["user_state"] = m.group(1).strip()
    else:
        m = re.search(r"<user_state>(.*?)(?=</?message>|$)", raw, re.DOTALL)
        if m and m.group(1).strip():
            result["user_state"] = m.group(1).strip()
        else:
            m = re.search(r"(#\s*User State Report.*?)(?=</?message>|\Z)", raw, re.DOTALL)
            if m and len(m.group(1).strip()) > 80:
                result["user_state"] = m.group(1).strip()

    m = re.search(r"<message>(.*?)</message>", raw, re.DOTALL)
    if m:
        msg = m.group(1).strip()
    else:
        m = re.search(r"<message>(.*?)$", raw, re.DOTALL)
        if m and m.group(1).strip():
            msg = m.group(1).strip()
        elif "</user_state>" in raw:
            msg = raw.split("</user_state>")[-1].strip()
        else:
            msg = ""
    msg = _strip_tags(msg)
    msg, wants_to_end = _extract_end_signal(msg)
    result["message"] = msg
    result["wants_to_end"] = wants_to_end
    return result
