"""Training utilities for S³-Sim.

Two helpers ship with the release:

* `assemble_sft` — concatenate one or more rollout-conversation directories
  and per-style QA-format JSONL files into a single multi-turn SFT JSONL
  consumable by TRL `SFTTrainer`, axolotl, llama-factory, etc.
* `configs/` — a reference LoRA-SFT YAML config for Qwen3 / Llama backbones.

The training step itself is intentionally external: the released code uses
TRL's `SFTTrainer` (or any equivalent multi-turn SFT runner) on the
assembled JSONL. See `training/README.md` for the full recipe.
"""
