"""Unit tests for run_eval_qa parsers + scorers (no LLM calls)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "run_eval_qa.py"
_spec = importlib.util.spec_from_file_location("run_eval_qa", _PATH)
run_eval_qa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval_qa)


def test_mcq_extract_final_answer():
    assert run_eval_qa.extract_mcq_letter("Some reasoning. Final Answer: B") == "B"
    assert run_eval_qa.extract_mcq_letter("final answer: c") == "C"

    assert run_eval_qa.extract_mcq_letter("Final Answer: [D]") == "D"


def test_mcq_extract_boxed():
    assert run_eval_qa.extract_mcq_letter("blah \\boxed{A}") == "A"
    assert run_eval_qa.extract_mcq_letter("$\\boxed{B}$") == "B"


def test_mcq_extract_strips_thinking():
    txt = "<think>weighing options...</think>Final Answer: C"
    assert run_eval_qa.extract_mcq_letter(txt) == "C"


def test_mcq_extract_returns_none_on_garbage():
    assert run_eval_qa.extract_mcq_letter("I don't know.") is None
    assert run_eval_qa.extract_mcq_letter("") is None


def test_mcq_extract_markdown_emphasis():
    """Larger models often markdown-bold the answer; the regex must accept it."""
    assert run_eval_qa.extract_mcq_letter("Reasoning. **Final Answer: B**") == "B"
    assert run_eval_qa.extract_mcq_letter("**Final Answer:** **C**") == "C"
    assert run_eval_qa.extract_mcq_letter("Final Answer:**A**") == "A"
    assert run_eval_qa.extract_mcq_letter("The correct answer is **D**.") == "D"
    assert run_eval_qa.extract_mcq_letter("I would pick option **B** because...") == "B"


def test_mcq_extract_display_math_boxed():
    assert run_eval_qa.extract_mcq_letter("$$\\boxed{C}$$") == "C"


def test_bigtom_substring_a():
    assert run_eval_qa.score_bigtom("Answer:a)Run the analysis", "a", "Run the analysis") is True


def test_bigtom_substring_b():
    assert run_eval_qa.score_bigtom("Answer:b)pick cheap", "b", "pick cheap") is True


def test_bigtom_other_letter_fails():
    assert run_eval_qa.score_bigtom("Answer:b)wrong", "a", "Run the analysis") is False


def test_bigtom_content_match_fallback():

    assert (
        run_eval_qa.score_bigtom(
            "I think the user wants to run the analysis with AASHTO.",
            "a",
            "Run the analysis with AASHTO damping",
        )
        is True
    )


def test_lamp_exact_match():
    assert run_eval_qa.score_lamp("structural_engineering", "structural_engineering") is True


def test_lamp_case_insensitive():
    assert run_eval_qa.score_lamp("Structural_Engineering", "structural_engineering") is True


def test_lamp_substring_match():
    assert (
        run_eval_qa.score_lamp("The category is structural_engineering.", "structural_engineering")
        is True
    )


def test_lamp_token_overlap():

    assert (
        run_eval_qa.score_lamp(
            "engineers value structural soundness and engineering rigor",
            "structural engineering",
        )
        is True
    )


def test_lamp_low_overlap_fails():
    assert run_eval_qa.score_lamp("totally unrelated", "structural engineering") is False


def test_lamp_underscore_label_token_match():
    """Snake_case labels should split on underscore for the overlap rule."""

    assert (
        run_eval_qa.score_lamp(
            "I would categorize this as personal finance and management",
            "financial_management",
        )
        is True
    )

    assert run_eval_qa.score_lamp("real estate", "financial_management") is False


def test_lamp_paraphrase_passes_at_50_percent_threshold():
    """`Astronomy Research` ↔ `Dark Matter Research` shares 1/2 tokens → pass."""
    assert run_eval_qa.score_lamp("Dark Matter Research", "Astronomy Research") is True


def test_resolve_max_tokens_default():
    assert run_eval_qa._resolve_max_tokens("gpt-4o-mini", 2048, {}) == 2048


def test_resolve_max_tokens_gpt5_family_auto_bumps():

    assert run_eval_qa._resolve_max_tokens("gpt-5-mini", 2048, {}) == 16384
    assert run_eval_qa._resolve_max_tokens("gpt-5-nano", 1024, {}) == 16384

    assert run_eval_qa._resolve_max_tokens("gpt-5-mini", 32000, {}) == 32000


def test_resolve_max_tokens_explicit_override_wins():
    overrides = {"gpt-5-mini": 8192, "gpt-4o": 4096}
    assert run_eval_qa._resolve_max_tokens("gpt-5-mini", 2048, overrides) == 8192
    assert run_eval_qa._resolve_max_tokens("gpt-4o", 2048, overrides) == 4096

    assert run_eval_qa._resolve_max_tokens("gpt-4o-mini", 2048, overrides) == 2048
