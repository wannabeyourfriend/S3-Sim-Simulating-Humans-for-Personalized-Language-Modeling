# Mind2Dialogue: State-Aware User Simulation for Theory-of-Mind and Personalization

[![Paper](https://img.shields.io/badge/arXiv-2512.06688-b31b1b.svg)](https://arxiv.org/abs/xxxx)
[![Dataset](https://img.shields.io/badge/HuggingFace-PersonaMem--v2-ffd21e.svg)](https://huggingface.co/datasets/wannabeyourfriend-hf/mind2dialogue)

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
├── user_simulator/                 
│   ├── oracle.py                                          
│   ├── prompts/
│   ├── simulator/               
│   │   ├── rollout.py           
│   │   ├── user_turn.py         
│   │   ├── parsing.py           
│   │   ├── persona_block.py     
│   │   └── behavior/         
│   └── qc/
├── training/                    # unsloth sft_trainer submodule
│   ├── sft_trainer.py           
│   ├── configs/                 
│   └── scripts/                 
├── evaluations/                 # personalization benchmarks collection submodule
│   └── multibench/benchmarks/   
├── data/                        # artifacts data for conversational rollouts
│   ├── filterd_refined_profiles/
│   ├── initial_prompts/
│   ├── rewritten_prompts/
│   └── behavior_modes/
├── run_rollout.py               
├── run_deep_scenario_rollout.py
├── run_qc.py
├── run_qa_construction.py
├── run_qa_rewrite.py            
└── run_eval_qa.py            
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
