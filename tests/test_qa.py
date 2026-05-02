"""Smoke tests for `user_simulator.qa`.

Each builder is exercised with a `FakeLLM` returning a scripted JSON.
The SFT line renderer is checked for byte-shape conformance with the
`build_sft_instance` schema (messages + metadata).
"""

from __future__ import annotations

import json

import pytest

from tests.fakes import FakeLLM
from user_simulator.data import Persona
from user_simulator.qa import (
    QAStyle,
    generate_for_conv,
    qa_item_to_sft_line,
    self_consistency_check_mcq,
)


def _persona() -> Persona:
    return Persona(
        id="p1",
        attributes={"age": 39, "occupation": "civil engineer", "country": "US"},
        summary="Senior civil engineer.",
        metadata={
            "refined_summary": "I am a 39-year-old civil engineer.",
            "behavioral_metadata": {
                "tone_pref": "formal",
                "expertise_level": "expert",
                "primary_domain": "civil_engineering",
            },
        },
    )


def _session() -> dict:
    return {
        "persona_id": "p1",
        "prompt_id": "scn_0",
        "conversation": [
            {"role": "user", "content": "I'm planning a 120-meter bridge in a seismic zone."},
            {
                "role": "assistant",
                "content": "Could you tell me about the soil class and target damping?",
            },
            {"role": "user", "content": "Soil class C, target 5% damping."},
            {
                "role": "assistant",
                "content": "Then I'd recommend a response-spectrum analysis with...",
            },
        ],
        "user_state_trajectory": [
            {
                "turn": 1,
                "user_state": "User is concerned about regulatory compliance and seismic safety.",
            },
            {
                "turn": 2,
                "user_state": "User has specified soil C and 5% damping; values precision.",
            },
            {"turn": 3, "user_state": "User wants concrete code references."},
        ],
        "ablation": "full",
        "profile_summary": "I am a 39-year-old civil engineer.",
        "behavioral_metadata": {"tone_pref": "formal"},
    }


@pytest.mark.asyncio
async def test_personamem_mcq_builder_and_renderer():
    fake = FakeLLM()
    fake.queue(
        "qa_personamem",
        json.dumps(
            {
                "ground_truth_preference": "User values precise, code-grounded recommendations",
                "user_query": "Which design approach should I pursue for the next phase?",
                "correct": "Use a response-spectrum analysis grounded in AASHTO LRFD with site-specific damping.",
                "stereotypical": "Go with the standard heuristics most engineers use.",
                "random": "Try a low-fat version of your favorite recipe.",
                "generic": "Consult relevant guidelines and consider a phased approach.",
            }
        ),
    )

    fake.queue(
        "qa_mcq_reasoning",
        "The user is a civil engineer who values precise, code-grounded "
        "advice. Option referencing AASHTO matches that. Final Answer: A",
    )
    item = await generate_for_conv(_persona(), _session(), QAStyle.PERSONAMEM_MCQ, fake)
    assert item is not None
    assert item.style is QAStyle.PERSONAMEM_MCQ
    assert item.options is not None and len(item.options) == 4
    assert item.correct_letter in ("A", "B", "C", "D")

    assert item.extra.get("has_cot") is True
    assert "Final Answer:" in item.answer_text

    assert item.answer_text.rstrip().endswith(f"Final Answer: {item.correct_letter}")

    line = qa_item_to_sft_line(item, _session())
    msgs = line["messages"]
    assert msgs[0]["role"] == "system"
    assert "<persona>" in msgs[0]["content"]

    assert msgs[-1] == {"role": "assistant", "content": item.answer_text}

    user_msg = msgs[-2]
    assert user_msg["role"] == "user"
    assert "Please choose the best answer" in user_msg["content"]
    assert "Please recall my related preferences" in user_msg["content"]
    assert "Final Answer: [Letter]" in user_msg["content"]

    assert line["metadata"]["qa_style"] == "personamem_mcq"
    assert line["metadata"]["source"] == "qa"
    assert line["metadata"]["correct_letter"] == item.correct_letter


@pytest.mark.asyncio
async def test_personamem_mcq_deterministic_shuffle():
    """Same (persona, scenario) → same letter assignment."""
    fake1, fake2 = FakeLLM(), FakeLLM()
    payload = json.dumps(
        {
            "ground_truth_preference": "x",
            "user_query": "q",
            "correct": "C",
            "stereotypical": "S",
            "random": "R",
            "generic": "G",
        }
    )
    fake1.queue("qa_personamem", payload)
    fake2.queue("qa_personamem", payload)
    fake1.queue("qa_mcq_reasoning", "reasoning. Final Answer: A")
    fake2.queue("qa_mcq_reasoning", "reasoning. Final Answer: A")
    item1 = await generate_for_conv(_persona(), _session(), QAStyle.PERSONAMEM_MCQ, fake1)
    item2 = await generate_for_conv(_persona(), _session(), QAStyle.PERSONAMEM_MCQ, fake2)
    assert item1.correct_letter == item2.correct_letter
    assert item1.options == item2.options


@pytest.mark.asyncio
async def test_personamem_mcq_appends_final_answer_if_missing():
    """If the reasoning model forgets the final-answer line, we append it."""
    fake = FakeLLM()
    fake.queue(
        "qa_personamem",
        json.dumps(
            {
                "ground_truth_preference": "x",
                "user_query": "q",
                "correct": "C",
                "stereotypical": "S",
                "random": "R",
                "generic": "G",
            }
        ),
    )
    fake.queue("qa_mcq_reasoning", "Just some reasoning without the canonical final-answer line.")
    item = await generate_for_conv(_persona(), _session(), QAStyle.PERSONAMEM_MCQ, fake)
    assert item is not None
    assert f"Final Answer: {item.correct_letter}" in item.answer_text


@pytest.mark.asyncio
async def test_personamem_mcq_falls_back_when_reasoning_call_fails():
    """If reasoning generation raises, fall back to the plain final-answer."""
    fake = FakeLLM()
    fake.queue(
        "qa_personamem",
        json.dumps(
            {
                "ground_truth_preference": "x",
                "user_query": "q",
                "correct": "C",
                "stereotypical": "S",
                "random": "R",
                "generic": "G",
            }
        ),
    )

    item = await generate_for_conv(_persona(), _session(), QAStyle.PERSONAMEM_MCQ, fake)
    assert item is not None
    assert item.extra.get("has_cot") is False
    assert item.answer_text == f"Final Answer: {item.correct_letter}"


@pytest.mark.asyncio
async def test_personamem_mcq_missing_keys_returns_none():
    fake = FakeLLM()
    fake.queue("qa_personamem", json.dumps({"user_query": "q", "correct": "x"}))
    item = await generate_for_conv(_persona(), _session(), QAStyle.PERSONAMEM_MCQ, fake)
    assert item is None


@pytest.mark.asyncio
async def test_prefeval_gen_builder_and_renderer():
    fake = FakeLLM()
    fake.queue(
        "qa_prefeval",
        json.dumps(
            {
                "preference": "I prefer designs that comply with AASHTO LRFD and dislike vague heuristics.",
                "question": "What's a good first step for the seismic analysis?",
                "assistant_response": "Since you prefer AASHTO-LRFD-compliant approaches, start with a response-spectrum analysis using the site-specific spectrum. Pin your damping at 5% as you specified, then validate with a time-history check on the critical mode.",
                "acknowledge_quote": "Since you prefer AASHTO-LRFD-compliant approaches",
            }
        ),
    )
    item = await generate_for_conv(_persona(), _session(), QAStyle.PREFEVAL_GEN, fake)
    assert item is not None
    line = qa_item_to_sft_line(item, _session())
    msgs = line["messages"]

    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant"]
    assert "AASHTO" in msgs[1]["content"]
    assert msgs[-1]["content"] == item.answer_text
    assert line["metadata"]["qa_style"] == "prefeval_gen"
    assert "AASHTO" in line["metadata"]["preference"]


@pytest.mark.asyncio
async def test_prefeval_drops_when_response_does_not_acknowledge():
    """Response with no overlap to preference and bogus ack_quote → drop."""
    fake = FakeLLM()
    fake.queue(
        "qa_prefeval",
        json.dumps(
            {
                "preference": "I strongly prefer plant-based meals and dislike anything containing peanuts.",
                "question": "What should I cook for dinner tonight?",
                "assistant_response": "Try grilled salmon with rice and a simple salad — it cooks fast and is satisfying.",
                "acknowledge_quote": "since you prefer plant-based",
            }
        ),
    )
    item = await generate_for_conv(_persona(), _session(), QAStyle.PREFEVAL_GEN, fake)
    assert item is None


@pytest.mark.asyncio
async def test_prefeval_keeps_via_token_overlap_fallback():
    """When ack_quote is wrong but the response shares ≥2 tokens (≥4 chars) with the preference, keep."""
    fake = FakeLLM()
    fake.queue(
        "qa_prefeval",
        json.dumps(
            {
                "preference": "I strongly prefer plant-based meals and dislike anything containing peanuts.",
                "question": "What should I cook for dinner tonight?",
                "assistant_response": "For a plant-based dinner avoiding peanuts, try a chickpea curry with basmati rice.",
                "acknowledge_quote": "this is a fabricated quote",
            }
        ),
    )
    item = await generate_for_conv(_persona(), _session(), QAStyle.PREFEVAL_GEN, fake)

    assert item is not None


@pytest.mark.asyncio
async def test_bigtom_builder_and_renderer():
    fake = FakeLLM()
    fake.queue(
        "qa_bigtom",
        json.dumps(
            {
                "narrative": "The user is reviewing two seismic-design options. Their conversation focused on AASHTO compliance.",
                "question": "What does the user most want to do next?",
                "option_a": "Run a response-spectrum analysis with AASHTO damping values.",
                "option_b": "Pick the cheapest option without analysis.",
                "correct_letter": "a",
            }
        ),
    )
    fake.queue(
        "qa_mcq_reasoning",
        "The user has been focused on AASHTO compliance throughout. "
        "Answer:a)Run a response-spectrum analysis with AASHTO damping values.",
    )
    item = await generate_for_conv(_persona(), _session(), QAStyle.BIGTOM_TOM, fake)
    assert item is not None
    assert item.correct_letter == "a"
    assert item.extra.get("has_cot") is True

    assert "Answer:a)" in item.answer_text

    line = qa_item_to_sft_line(item, _session())
    msgs = line["messages"]
    assert "Answer as 'Answer:<option>)<answer>'" in msgs[0]["content"]
    user_msg = msgs[1]["content"]
    assert "a) " in user_msg and "b) " in user_msg
    assert msgs[2] == {"role": "assistant", "content": item.answer_text}


@pytest.mark.asyncio
async def test_bigtom_invalid_letter_returns_none():
    fake = FakeLLM()
    fake.queue(
        "qa_bigtom",
        json.dumps(
            {
                "narrative": "x",
                "question": "y",
                "option_a": "p",
                "option_b": "q",
                "correct_letter": "z",
            }
        ),
    )
    item = await generate_for_conv(_persona(), _session(), QAStyle.BIGTOM_TOM, fake)
    assert item is None


@pytest.mark.asyncio
async def test_lamp_cls_builder_and_renderer():
    fake = FakeLLM()
    fake.queue(
        "qa_lamp",
        json.dumps(
            {
                "task_family": "tag_classify",
                "profile_items": [
                    {
                        "input": "Cantilever bridge load distribution",
                        "output": "structural_engineering",
                    },
                    {"input": "AASHTO LRFD code update", "output": "code_compliance"},
                    {"input": "Seismic retrofit case study", "output": "structural_engineering"},
                ],
                "query": "Damping coefficient calibration for tall piers",
                "target": "structural_engineering",
            }
        ),
    )
    item = await generate_for_conv(_persona(), _session(), QAStyle.LAMP_CLS, fake)
    assert item is not None
    assert item.answer_text == "structural_engineering"

    line = qa_item_to_sft_line(item, _session())
    msgs = line["messages"]
    assert "past items" in msgs[1]["content"]
    assert "Damping coefficient" in msgs[1]["content"]
    assert msgs[-1] == {"role": "assistant", "content": "structural_engineering"}
    assert line["metadata"]["task_family"] == "tag_classify"


@pytest.mark.asyncio
async def test_lamp_too_few_items_returns_none():
    fake = FakeLLM()
    fake.queue(
        "qa_lamp",
        json.dumps(
            {
                "task_family": "tag_classify",
                "profile_items": [{"input": "x", "output": "y"}],
                "query": "q",
                "target": "t",
            }
        ),
    )
    item = await generate_for_conv(_persona(), _session(), QAStyle.LAMP_CLS, fake)
    assert item is None


@pytest.mark.asyncio
async def test_self_consistency_drops_easy_mcq():
    """If profile-stripped model gets the right letter every time, drop."""
    fake = FakeLLM()
    payload = json.dumps(
        {
            "ground_truth_preference": "x",
            "user_query": "q",
            "correct": "C",
            "stereotypical": "S",
            "random": "R",
            "generic": "G",
        }
    )
    fake.queue("qa_personamem", payload)
    fake.queue("qa_mcq_reasoning", "reasoning. Final Answer: A")
    item = await generate_for_conv(_persona(), _session(), QAStyle.PERSONAMEM_MCQ, fake)
    assert item is not None
    target = item.correct_letter

    for _ in range(3):
        fake.queue("qa_self_consistency", f"Reasoning... Final Answer: {target}")
    assert await self_consistency_check_mcq(item, fake) is False


@pytest.mark.asyncio
async def test_self_consistency_keeps_hard_mcq():
    """If profile-stripped model never gets it right, keep."""
    fake = FakeLLM()
    payload = json.dumps(
        {
            "ground_truth_preference": "x",
            "user_query": "q",
            "correct": "C",
            "stereotypical": "S",
            "random": "R",
            "generic": "G",
        }
    )
    fake.queue("qa_personamem", payload)
    fake.queue("qa_mcq_reasoning", "reasoning. Final Answer: A")
    item = await generate_for_conv(_persona(), _session(), QAStyle.PERSONAMEM_MCQ, fake)
    assert item is not None
    target = item.correct_letter
    wrong = next(L for L in "ABCD" if L != target)
    for _ in range(3):
        fake.queue("qa_self_consistency", f"Final Answer: {wrong}")
    assert await self_consistency_check_mcq(item, fake) is True
