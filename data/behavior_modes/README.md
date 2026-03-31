# User Simulator Behavior Prompts Library
This file contains the Taxonomy of User Needs and Actions (Authors: Renée Shelby, Fernando Diaz, Vinodkumar Prabhakaran,2025) from 1193 WildChat and ShareGPT logs to guide user simulator behavior.[arXiv:2510.06124 [cs.HC]]

## Fields
| Field | Type | Required | Description |
|---|---|---|---|
| `behavior_id` | string | yes | Unique identifier, e.g. `analysis`, `clarification`, `default_behavior` |
| `name` | string | yes | Human-readable name, e.g. `Analysis`, `Clarification` |
| `tuna_mode` | string | yes | Parent mode from TUNA taxonomy, e.g. `Information Processing & Synthesis`, `Mixed` |
| `tuna_strategy` | string | no | Specific strategy within the TUNA mode, e.g. `Analysis`, `Clarification` |
| `cognitive_delegation_level` | string | no | How much cognitive work is delegated to the AI, e.g. `Low`, `Medium`, `Medium to High`, `High` |
| `description` | string (multiline) | yes | Explains the behavior's purpose, cognitive framing, and theoretical grounding |
| `guidance_template` | string (multiline) | yes | Injected into the user simulator prompt to steer turn generation. Contains request type options, authenticity rules, and an internal question |
| `few_shot_examples` | list of objects | no | Example user turns demonstrating the behavior (absent in `default_behavior`) |

### `few_shot_examples[]` fields

| Field | Type | Description |
|---|---|---|
| `request_type` | string | Sub-category of the behavior, e.g. `comparative_analysis`, `explanation_request` |
| `user_turn` | string | Example user message demonstrating this request type |

## 3-Level Hierarchy
1. 6 Behavior Mode of user intent  
2. 14 Strategies of how to achieve it  
3. 57 Request Types of atomic action

```txt
   ┌─────────────────────────────────────────────────────────┐
   │  Mode 6: Meta-Conversation  (contextual / management)   │
   │  ┌───────────────────────────────────────────────────┐  │
   │  │  Mode 5: Social Interaction  (relational layer)   │  │
   │  │  ┌─────────────────────────────────────────────┐  │  │
   │  │  │  Instrumental Core (Modes 1–4)              │  │  │
   │  │  │  • Mode 1: Information Seeking              │  │  │
   │  │  │  • Mode 2: Information Processing           │  │  │
   │  │  │  • Mode 3: Procedural Guidance & Execution  │  │  │
   │  │  │  • Mode 4: Content Creation & Transformation│  │  │
   │  │  └─────────────────────────────────────────────┘  │  │
   │  └───────────────────────────────────────────────────┘  │
   └─────────────────────────────────────────────────────────┘
```

- **6** Meta-Conversation (context / mgmt)  
- **5** Social Interaction (relational)  
- **1-4** Instrumental Core  
  - **1** Information Seeking  
  - **2** Information Processing  
  - **3** Procedural Guidance & Execution  
  - **4** Content Creation & Transformation  

## Design Rule
Users improvise moment-to-moment; simulator must reflect this, not scripted flows.