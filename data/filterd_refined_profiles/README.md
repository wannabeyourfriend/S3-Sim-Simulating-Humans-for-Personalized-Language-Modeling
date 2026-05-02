# summary_refined_profiles_{us,cn,de,in,jp}.jsonl

Five JSONL files — one per country — each with one persona per line:

| File | Personas |
|---|---:|
| `summary_refined_profiles_us.jsonl` | 61 |
| `summary_refined_profiles_cn.jsonl` | 58 |
| `summary_refined_profiles_de.jsonl` | 59 |
| `summary_refined_profiles_in.jsonl` | 56 |
| `summary_refined_profiles_jp.jsonl` | 57 |

Every persona carries the same field schema documented below, regardless
of country. `persona_id` namespaces are independent across files.

## Fields

| Field | Type | Description |
|---|---|---|
| `persona_id` | string | Unique profile identifier, e.g. `profile_259` |
| `summary` | string | Original free-text persona summary |
| `refined_summary` | string | LLM-refined persona summary with sharper detail |
| `behavioral_metadata` | object | Structured behavioral attributes |

### `behavioral_metadata` keys

| Key | Example values |
|---|---|
| `age_band` | `25-34`, `35-44`, `45-54` |
| `specialization` | `civil_infrastructure_engineering` |
| `long_term_goal` | free-text string |
| `tone_pref` | `neutral`, `casual`, `formal` |
| `multi_intent_rate` | `occasional_multi`, `frequent_multi`, `single_intent` |
| `query_length_pref` | `short`, `medium`, `long` |
| `ai_familiarity` | `comfortable`, `cautious`, `enthusiastic` |
| `interaction_frequency` | `regular`, `occasional`, `frequent` |
| `follow_up_tendency` | `deep_dive`, `surface_level`, `moderate` |
| `creativity_vs_precision` | `precision_focused`, `balanced`, `creativity_leaning` |
| `time_sensitivity` | `somewhat_urgent`, `relaxed`, `highly_urgent` |
| `collaboration_context` | `small_team`, `solo`, `large_org` |
| `error_tolerance` | `zero_tolerance`, `moderate`, `high` |
| `feedback_style` | `direct_correction`, `gentle_suggestion`, `detailed_critique` |
| `context_retention_expectation` | `within_session`, `across_sessions` |
| `autonomy_level` | `moderate_autonomy`, `high_autonomy`, `guided` |
| `detail_orientation` | `moderate_detail`, `high_detail`, `low_detail` |
| `explanation_depth` | `step_by_step`, `overview_first`, `concise` |
| `primary_domain` | `engineering`, `education`, `healthcare`, etc. |
| `typical_scenario` | `problem_solving`, `learning`, `creative_task`, etc. |