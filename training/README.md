# Training

S³-Sim conversational data is consumed by any standard multi-turn SFT
trainer (TRL `SFTTrainer`, axolotl, llama-factory, …). The trainer is
intentionally external to keep the released code minimal.

## Recipe (one-liner)

```bash
# 1. Generate data
python run_rollout.py            --ablation full --concurrency 80
python run_qc.py                 --conversations-dir output/conversations/full
python run_qa_construction.py    --conversations-dir output/conversations/full \
                                 --qc-results        output/qc/v1/qc_results.jsonl \
                                 --output-dir        output/qa/v1

# 2. Assemble SFT
python -m training.assemble_sft \
       --conv-dirs output/conversations/full \
       --qa-files  output/qa/v1/personamem_mcq.jsonl \
                   output/qa/v1/prefeval_gen.jsonl \
       --output    data/sft/train.jsonl

# 3. Train
accelerate launch -m trl sft \
       --model_name_or_path Qwen/Qwen3-8B \
       --dataset_name       data/sft/train.jsonl \
       --use_peft --lora_r 16 --lora_alpha 32 \
       --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
       --learning_rate 2e-4 --num_train_epochs 3 \
       --max_seq_length 4096 --packing True --bf16 \
       --output_dir output/sft/qwen3_8b_s3sim
```

Full hyperparameters and reasoning live in
`training/configs/lora_qwen3_8b.yaml`.

## Data shape

Every line of the assembled JSONL has the OpenAI / TRL chat schema:

```json
{
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "persona_id": "...", "scenario_id": "...",
    "ablation":   "...", "source":      "qa" | "rollout"
  }
}
```

Conversation-source lines come from `user_simulator.oracle.assemble_sft`;
QA-source lines come from `user_simulator.qa.qa_item_to_sft_line`. Both
share the same line schema, so the trainer is fully format-agnostic.

## Hardware

Reference run: 4 × A100 80 GB, ≈ 4 h for ~35 k samples × 3 epochs at
4 k context. The recipe is single-GPU compatible — drop
`gradient_accumulation_steps` to keep the effective batch size and use
`accelerate config` with one device.
