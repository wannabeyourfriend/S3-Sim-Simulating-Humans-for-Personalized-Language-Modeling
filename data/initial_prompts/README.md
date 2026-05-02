# prompts_mixed_taged.jsonl

JSONL file, one line per prompt (39 896 total). Sourced from public
multi-domain instruction collections (Artificial Hivemind, OpenR1-Math,
ShareGPT, WildChat, …) and lightly tagged with a domain/register/task
fingerprint by an LLM.

This is the upstream pool that feeds the rewritten persona-grounded
prompts under `../rewritten_prompts/`.

## Fields

| Field | Type | Description |
|---|---|---|
| `prompt_id` | string | Source-anchored identifier, e.g. `Artificial_Hivemind_66269` |
| `prompt_text` | string | Verbatim user prompt |
| `fingerprint` | object | LLM-derived prompt-shape tags |

### `fingerprint` keys

| Key | Example values |
|---|---|
| `domain` | `["creative"]`, `["finance"]`, `["coding", "education"]` |
| `expertise_level_implied` | `""`, `"novice"`, `"expert"` |
| `register` | `"casual"`, `"formal"`, `"technical"` |
| `region` | `["GLOBAL"]`, `["US"]`, `["CN", "JP"]` |
| `task_type` | `"creative"`, `"analytical"`, `"procedural"`, … |
| `prompt_complexity` | `"simple"`, `"medium"`, `"complex"` |
| `_meta.confidence` | `"high"`, `"medium"`, `"low"` |
| `_meta.inferred_fields` | list of field names whose value was inferred (vs. extracted) |
