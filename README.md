# Mind2Dialogue: State-Aware User Simulation for Theory-of-Mind and Personalization

🚨 **[Abstract]** Developing Large Language Models capable of true personalization remains a significant challenge, primarily due to the scarcity of high-quality, private personalized conversation data. Existing approaches rely on publicly available internet data, which suffers from
severe distribution shifts, or synthetic data generated from static personas, which fail to capture the dynamic causal structure of
real-world interactions. We propose a novel data generation pipeline
that synthesizes conversation trajectories by explicitly and
structurally maintaining latent user state, encompassing psychological
dynamics, beliefs, and evolving social relationships. Unlike naïve
scaling strategies, our approach uses these evolving user states as continuous constraints to guide generation, ensuring that the synthesized dialogues reflect realistic causal depth rather than
surface-level mimicry. Experiments show that models fine-tuned on our data exhibit superior sample efficiency and significantly improved capabilities in intention inference and theory-of-mind reasoning compared to baselines.

### Repository layout

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
├── evaluations/                 # six personalization benchmarks (submodule)
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

### Installation

```bash
git clone --recursive https://github.com/wannabeyourfriend/mind2dialogue.git
cd mind2dialogue
uv sync  # or: pip install -e .
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

Generate conversational rollouts.

```bash
python run_rollout.py --ablation full --concurrency 80
```

Or scenario-driven rollouts (lifelong / highfreq / concerning / affective):

```bash
python run_deep_scenario_rollout.py --constructor simulator_lifelong_scenario_constructor --concurrency 80
```

```bash
python run_qc.py --conversations-dir output/conversations/full --output-dir output/qc/v1
```

Construct SFT sample by using user state aware minds in conversations as ground-truth.

```bash
python run_qa_construction.py --conversations-dir output/conversations/full --qc-results output/qc/v1/qc_results.jsonl --output-dir output/qa/v1
```

Run any benchmark

```bash
git submodule update --init evaluations
pip install -e evaluations
multibench run personamem -- --api-base <server_endpoint> --model <run_name> --workers 64 --output-dir results/<run_name>/PersonaMem
```

### Citation

```bibtex
@article{mind2dialogue2026,
  title  = {Mind2Dialogue: State-Aware User Simulation for Theory-of-Mind and Personalization},
  author = {Anonymous},
  year   = {2026},
}
```

### Responsible use

The release ships a **concerning-scenario** generator (`simulator_concerning_scenario_constructor`) that produces sensitive prompts wrapped in legitimising personas. These are intended **only** for training and benchmarking refusal-quality and safe-completion behavior; they are clearly tagged in `metadata.scenario_family = "concerning"` so downstream consumers can opt out. Do not use the released data to fine-tune an assistant that lacks an upstream safety layer.

License: MIT (code) · CC-BY-4.0 (data artifacts under `data/`).
