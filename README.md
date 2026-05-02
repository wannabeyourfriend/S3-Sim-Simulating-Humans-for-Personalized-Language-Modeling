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

## Method at a glance

Generation is a stateful POMDP. At each turn $t$ the simulator carries a
user state $s_t = \langle U,\; C,\; B_t \rangle$:

| Component | Meaning | Refresh |
|-----------|---------|---------|
| **U**     | Immutable persona profile (demographics, traits, behavioral metadata) | once, per persona |
| **C**     | Conversational scenario / functional context | once, per session |
| **B_t**   | Dynamic behavioral mode (epistemic deepening, self-disclosure, …) | sampled every turn |

Utterances follow a two-stage causal chain $U \to B_t \to m_t$ that
decouples strategic intent from surface tokens. A privileged
**oracle assistant** with full visibility into $s_t$ generates expert
responses; the student is fine-tuned via SFT on observable history
$H_{<t}$ alone, minimising

$$
\mathcal{L}(\phi) \;=\; -\,\mathbb{E}_{(H_{<t},\,a_t)\sim\mathcal{D}}\bigl[\log P_\phi(a_t\mid H_{<t})\bigr]
$$

— forcing the student to *implicitly reconstruct* latent dynamics.

```text
                ┌──────────────────────────────────┐
                │        Persona library U         │
                └───────────────┬──────────────────┘
                                ▼
   ┌──────────────────────────────────────────────────────┐
   │  Behavior controller          (LLM, sampled per turn)│
   │  ─────────────────────────────────────────────────── │
   │  picks B_t ∈ behavior library given persona, state,  │
   │  conversation prefix and previous behavior history   │
   └────────┬───────────────────────────┬─────────────────┘
            ▼                           ▼
   ┌────────────────────┐     ┌────────────────────────┐
   │  User simulator    │ ⇄   │  Oracle assistant      │
   │  (state + B_t)     │     │  (profile + state)     │
   └────────────────────┘     └────────────────────────┘
            │                           │
            └────────────┬──────────────┘
                         ▼
                ┌──────────────────┐
                │  Multi-turn JSON │  →  QC tiers (D1–D6)  →  SFT JSONL
                └──────────────────┘
```

The five canonical ablations decompose the contribution of each axis:

| Ablation              | User simulator        | Assistant strategy        | What it isolates              |
|-----------------------|-----------------------|---------------------------|-------------------------------|
| `full`                | state + behavior      | oracle (profile + state)  | full pipeline (release default) |
| `no_privilege`        | state + behavior      | vanilla (no profile)      | privileged distillation gain  |
| `no_behavior`         | state only            | oracle (profile + state)  | dynamic behavior contribution |
| `no_state`            | vanilla               | oracle (profile + state)  | latent state contribution     |
| `oracle_profile_only` | state + behavior      | oracle (profile only)     | state-channel contribution    |

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
├── training/                    # SFT assembly + reference LoRA recipe
│   ├── assemble_sft.py          # concatenate conv + QA JSONLs → train.jsonl
│   ├── configs/                 # reference TRL/axolotl YAML
│   └── README.md                # one-liner training recipe
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

# 4. Assemble SFT mix (conversation + QA)
python -m training.assemble_sft \
       --conv-dirs output/conversations/full \
       --qa-files  output/qa/v1/personamem_mcq.jsonl \
                   output/qa/v1/prefeval_gen.jsonl \
       --output    data/sft/train.jsonl

# 5. Train (TRL SFTTrainer; full recipe in training/README.md)
accelerate launch -m trl sft \
       --model_name_or_path Qwen/Qwen3-8B \
       --dataset_name       data/sft/train.jsonl \
       --use_peft --lora_r 16 --lora_alpha 32 \
       --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
       --learning_rate 2e-4 --num_train_epochs 3 \
       --max_seq_length 4096 --packing True --bf16 \
       --output_dir output/sft/qwen3_8b_s3sim

# 6. Evaluate
python run_eval_qa.py --qa-dir output/qa/v1 \
                      --models output/sft/qwen3_8b_s3sim
python -m multibench --model output/sft/qwen3_8b_s3sim   # six benchmarks
```

Every entry script is **resumable** — each writes its outputs incrementally
(append-mode JSONL + per-conversation JSON files) and skips work that has
already completed.

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
schema — see `training/README.md` for the canonical line shape and
`samples/` for one-line examples of every output type.

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
