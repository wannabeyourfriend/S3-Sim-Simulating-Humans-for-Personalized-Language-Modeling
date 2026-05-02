# Released artifacts

| Subdirectory | Contents |
|---|---|
| `behavior_modes/` | 17 behavior YAMLs + the controller catalog. Loaded at import time by `user_simulator.simulator.behavior.library`. |
| `filterd_refined_profiles/` | Persona library. Five JSONL files, one per country (US, CN, DE, IN, JP), 56–61 personas each. |
| `initial_prompts/` | Upstream multi-source prompt pool (≈40 k prompts, lightly tagged). The raw input to persona-grounded prompt rewriting. |
| `rewritten_prompts/` | Persona-grounded rewrites used as rollout seeds (1 240 prompts: 62 US personas × 20 prompts each). |

Each subdirectory ships its own `README.md` with the full field schema.

The release does **not** ship pre-generated rollout conversations or
SFT JSONLs — those are large and expensive to regenerate consistently.
Run the pipeline (`run_rollout.py`, `run_qc.py`, `run_qa_construction.py`)
against this data to produce the training corpus from scratch.
