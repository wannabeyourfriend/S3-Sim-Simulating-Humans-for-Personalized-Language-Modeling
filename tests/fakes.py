"""Test doubles for `user_simulator.data.LLM`.

Two doubles are provided:

* `FakeLLM` — replays scripted responses keyed by `call_type`. Use for
  deterministic unit tests where you know which call types fire and in what
  order.
* `RecordingLLM` — wraps a real `LLM`, records every call to a JSONL file.
  Use once to capture a golden trace from the real backend, then feed the
  recording into `FakeLLM` for the regression gate.

Both implement the subset of the `LLM` interface that the rollout uses:
`chat()`, `chat_json()`, `model`, `stats`. They do NOT implement
the optional JSONL call logging — that's an orthogonal concern.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FakeLLM:
    """Replay scripted responses by `call_type`.

    Loads a JSONL file where each line is `{"call_type": ..., "content": ...}`.
    On each `chat(call_type=X)` invocation, pops the next response for that
    call_type from the queue. If the queue is empty, raises `RuntimeError`
    with the call signature so the test author knows what to script.

    Recognized call_type values used by the pipeline:
      - "chat"                 (default — assistant + vanilla user turns)
      - "behavior_controller"  (LLM behavior selector)
      - "scenario_constructor" (deep-scenario rollout only)
    User-simulator turns reuse the default "chat" call_type today.
    """

    script_path: Path | None = None
    model: str = "fake-llm"
    calls: int = 0
    tokens: int = 0
    _queues: dict[str, deque[dict]] = field(default_factory=lambda: defaultdict(deque))

    def __post_init__(self) -> None:
        if self.script_path is not None:
            self.load_script(self.script_path)

    def load_script(self, path: Path) -> None:
        """Load scripted responses from a JSONL file."""
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            call_type = rec.get("call_type", "chat")
            self._queues[call_type].append(rec)

    def queue(self, call_type: str, content: str, **extra: Any) -> None:
        """Append a single canned response programmatically."""
        self._queues[call_type].append({"call_type": call_type, "content": content, **extra})

    async def chat(
        self,
        messages,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        return_thinking: bool = False,
        call_type: str = "chat",
    ):
        q = self._queues.get(call_type)
        if not q:
            raise RuntimeError(
                f"FakeLLM: no scripted response for call_type={call_type!r}. "
                f"Queue contents: {dict((k, len(v)) for k, v in self._queues.items())}"
            )
        rec = q.popleft()
        content = rec["content"]
        self.calls += 1
        self.tokens += rec.get("tokens", 0)
        if return_thinking:
            return content, rec.get("thinking", "")
        return content

    async def chat_json(self, messages, **kw) -> dict:
        text = await self.chat(messages, json_mode=True, **kw)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            s, e = text.find("{"), text.rfind("}") + 1
            if s >= 0 and e > s:
                return json.loads(text[s:e])
            raise

    @property
    def stats(self) -> dict:
        return {"calls": self.calls, "tokens": self.tokens}


@dataclass
class RecordingLLM:
    """Wraps a real `LLM` and writes every call to a JSONL file.

    Use this once against the real backend to capture a deterministic golden
    trace, then feed `recording_path` into `FakeLLM(script_path=...)`.

    Usage:
        rec = RecordingLLM(real_llm, Path("tests/fixtures/llm_scripts/full.jsonl"))
        await rollout_conversation(persona, prompt, "p_0", rec, config=cfg)
    """

    inner: Any
    recording_path: Path

    def __post_init__(self) -> None:
        self.recording_path.parent.mkdir(parents=True, exist_ok=True)

        self.recording_path.write_text("", encoding="utf-8")

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def stats(self) -> dict:
        return self.inner.stats

    async def chat(self, messages, **kw):
        call_type = kw.get("call_type", "chat")
        result = await self.inner.chat(messages, **kw)
        if isinstance(result, tuple):
            content, thinking = result
            rec = {"call_type": call_type, "content": content, "thinking": thinking}
        else:
            content = result
            rec = {"call_type": call_type, "content": content}
        rec["timestamp"] = time.time()
        with open(self.recording_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return result

    async def chat_json(self, messages, **kw) -> dict:
        text = await self.chat(messages, json_mode=True, **kw)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            s, e = text.find("{"), text.rfind("}") + 1
            if s >= 0 and e > s:
                return json.loads(text[s:e])
            raise
