# S³-Sim: Simulating Humans for Personalized Language Modeling

> **Structured State Simulation** — a white-box user simulator + privileged
> distillation pipeline for generating multi-turn personalization data with
> evolving latent user state.

S³-Sim addresses the central bottleneck of personalized LLM alignment:
high-quality conversational data is private, while public synthetic data
uses static personas that capture surface behavior but miss the causal
link between a user's *internal* psychological state and their utterances.
S³-Sim closes this gap by maintaining an explicit, evolving latent state
during generation, then training a student model to reconstruct that
state implicitly from observable history alone.

---

## Method

Each turn, the simulator carries a latent user state of three parts:
an immutable **persona profile**, a fixed **scenario context**, and a
per-turn **behavioral mode** sampled by an LLM controller. Utterances
are produced as `persona → behavior → message`, so strategic intent
is decoupled from surface tokens.

A privileged **oracle assistant** with full visibility into the user
state generates the assistant turn. The student model is fine-tuned
via SFT on the observable conversation history alone — forcing it
to reconstruct latent user dynamics implicitly at inference time.

Five named ablations isolate the contribution of each axis:

| Ablation              | User simulator        | Assistant strategy        |
|-----------------------|-----------------------|---------------------------|
| `full`                | state + behavior      | oracle (profile + state)  |
| `no_privilege`        | state + behavior      | vanilla (no profile)      |
| `no_behavior`         | state only            | oracle (profile + state)  |
| `no_state`            | vanilla               | oracle (profile + state)  |
| `oracle_profile_only` | state + behavior      | oracle (profile only)     |

---

## Repository layout

```
.
├── user_simulator/              # core library — no I/O at import time
│   ├── ablation.py              # AblationConfig + 5 named presets
│   ├── data.py                  # LLM client, Persona/Scenario, conv I/O helpers
│   ├── oracle.py                # privileged annotation + SFT assembly
│   ├── sft.py                   # canonical SFT line builder
│   ├── qa.py                    # 4 personalized QA-style builders (PersonaMem/PrefEval/BigTom/LaMP)
│   ├── prompts/                 # YAML prompt templates (Jinja-style {{var}})
│   ├── simulator/               # multi-turn rollout
│   │   ├── rollout.py           # main async loop
│   │   ├── user_turn.py         # stateful & vanilla user-turn generators
│   │   ├── parsing.py           # output parsers for <user_state>/<message>
│   │   ├── persona_block.py     # persona → prompt-block helpers
│   │   └── behavior/            # behavior library, selection, block rendering
│   └── qc/                      # 6-dimension quality-check pipeline (D1–D6)
│
├── training/                    # SFT trainer + serving (submodule)
│   ├── sft_trainer.py           # single-file Unsloth + TRL multi-turn SFT
│   ├── configs/                 # one YAML per run
│   └── scripts/                 # train + vLLM serving launchers
│
├── p13n-eval-harness/           # six personalization benchmarks (submodule)
│   └── multibench/benchmarks/   # bigtom · lamp · personalens · personamem · prefeval · sotopia
│
├── data/                        # released artifacts
│   ├── filterd_refined_profiles/  # persona library (US, CN, DE, IN, JP)
│   ├── initial_prompts/           # raw real-query prompt pool
│   ├── rewritten_prompts/         # persona-grounded rewrites (rollout seeds)
│   └── behavior_modes/            # 17 behavior YAMLs + controller catalog
│
├── samples/                     # one-line examples of every output shape
│
├── run_rollout.py               # entry: rewritten prompts → conversations
├── run_deep_scenario_rollout.py # entry: lifelong/highfreq/concerning/affective scenarios
├── run_qc.py                    # entry: D1–D6 quality-check pipeline
├── run_qa_construction.py       # entry: conversations → 4 QA-style SFT JSONLs
├── run_qa_rewrite.py            # entry: v1 QA → v2 (harder, persona-grounded)
└── run_eval_qa.py               # entry: benchmark models on QA-format slices
```

---

## Installation

```bash
git clone --recursive https://github.com/wannabeyourfriend/S3-Sim-Simulating-Humans-for-Personalized-Language-Modeling.git
cd     S3-Sim-Simulating-Humans-for-Personalized-Language-Modeling
uv sync                                          # or: pip install -e .
```

Set OpenAI-compatible LLM credentials before running anything:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"   # or any compat. endpoint
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4o-mini"
```

The simulator and oracle accept independent model overrides via
`SIM_MODEL` / `ORACLE_MODEL` and a separate `JUDGE_MODEL` for the QC and
benchmark judges (mitigates self-judging bias).

---

## End-to-end pipeline

```bash
# 1. Generate rollouts (async; persona + ablation choices)
python run_rollout.py            --ablation full --concurrency 80
# or scenario-driven rollouts
python run_deep_scenario_rollout.py \
       --constructor simulator_lifelong_scenario_constructor \
       --concurrency 80

# 2. Quality-check (D1–D4 programmatic; D5–D6 LLM-judge)
python run_qc.py --conversations-dir output/conversations/full \
                 --output-dir        output/qc/v1

# 3. Construct QA-format SFT (PersonaMem/PrefEval/BigTom/LaMP)
python run_qa_construction.py \
       --conversations-dir output/conversations/full \
       --qc-results        output/qc/v1/qc_results.jsonl \
       --output-dir        output/qa/v1

# 4. Train (Unsloth + TRL multi-turn SFT; see training/ submodule)
git submodule update --init training
cp output/qa/v1/*.jsonl training/data/
bash training/scripts/train_qwen3_4b_modeB.sh    # or any configs/*.yaml

# 5. Evaluate
#    (a) in-distribution check on the QA slices we trained on
python run_eval_qa.py --qa-dir output/qa/v1 \
                      --models training/outputs/<run_name>
#    (b) six personalization benchmarks via p13n-eval-harness — serve first
bash training/scripts/serve_qwen3_4b_no_think.sh                # vLLM @ :8002
git submodule update --init p13n-eval-harness
pip install -e p13n-eval-harness
multibench run personamem  -- --api-base http://localhost:8002/v1 \
                              --model    <run_name> --workers 64 \
                              --output-dir results/<run_name>/PersonaMem
# repeat for: bigtom · lamp · personalens · prefeval · sotopia
```

Every entry script is **resumable** — each writes its outputs incrementally
(append-mode JSONL + per-conversation JSON files) and skips work that has
already completed.

---

## Evaluation portfolio

Two complementary eval paths ship with the release:

* `run_eval_qa.py` — in-distribution check on the four QA-format slices
  generated by `run_qa_construction.py`. Per-style answer extractors
  byte-match the upstream benchmarks' parsers.

* `p13n-eval-harness/` (submodule) — six end-to-end personalization
  benchmarks behind one CLI (`multibench run <bench> --`):

  | Benchmark     | Capability                          | Protocol            |
  |---------------|-------------------------------------|---------------------|
  | `personamem`  | long-term memory                    | implicit-persona MCQ|
  | `prefeval`    | preference adherence                | gen + cls + judge   |
  | `bigtom`      | theory of mind                      | ToM MCQ (2 × 200)   |
  | `lamp`        | per-task personalization            | 7 tasks (F1/MAE/BLEU)|
  | `personalens` | long-context personalized dialogue  | gen + 4-dim judge   |
  | `sotopia`     | social interaction                  | 7-dim social judge  |

  All six accept the same base flags (`--api-base · --model · --workers
  · --output-dir`); see `p13n-eval-harness/README.md` for per-benchmark
  options.

---

## Data shape

Every conversation JSON carries the full latent trajectory:

```json
{
  "persona_id": "profile_259",
  "prompt_id":  "us_profile_259_q3_rewritten",
  "ablation":   "full",
  "conversation": [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "user_state_trajectory":  [{"turn": 1, "user_state": "# User State Report ..."}],
  "behavior_trajectory":    [{"turn": 1, "behavior": "epistemic_deepening", ...}],
  "profile_summary":        "...",
  "behavioral_metadata":    {...},
  "num_turns": 12,
  "termination": "user_ended | max_turns | empty_message",
  "models":            {"user_simulator": "...", "assistant": "...", ...},
  "prompt_templates":  {...}
}
```

Both rollout-source and QA-source SFT lines share the OpenAI / TRL chat
schema — see `samples/` for one-line examples of every output type
and the `training/` submodule for the trainer that consumes them.

---

## Citation

```bibtex
@article{s3sim2026,
  title  = {{S\textsuperscript{3}-Sim}: Simulating Humans with Structured Latent State for Personalized Language Modeling},
  author = {Anonymous},
  year   = {2026},
  note   = {Submitted to NeurIPS Datasets \& Benchmarks Track}
}
```

---

## Responsible use

The release ships a **concerning-scenario** generator
(`simulator_concerning_scenario_constructor`) that produces sensitive
prompts wrapped in legitimising personas. These are intended **only** for
training and benchmarking refusal-quality and safe-completion behavior;
they are clearly tagged in `metadata.scenario_family = "concerning"` so
downstream consumers can opt in or out. Do not use the released data to
fine-tune an assistant that lacks an upstream safety layer.

License: MIT (code) · CC-BY-4.0 (data artifacts under `data/`).
