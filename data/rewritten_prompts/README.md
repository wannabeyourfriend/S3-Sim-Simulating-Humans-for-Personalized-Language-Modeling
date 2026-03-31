# original_rewritten_selected_prompts_us.jsonl

JSONL file, one line per prompt (1240 total, from 62 personas x 20 prompts each).

## Fields

| Field | Type | Description |
|---|---|---|
| `persona_id` | string | Profile this prompt belongs to, e.g. `profile_259` |
| `prompt_id` | string | Source dataset identifier, e.g. `open-r1/OpenR1-Math-220k_102002` |
| `original` | string | Original prompt text from source dataset |
| `rewritten` | string | Persona-voice rewritten prompt grounded in the profile's background and goals |