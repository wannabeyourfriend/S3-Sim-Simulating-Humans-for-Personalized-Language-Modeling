# User Simulator Behavior Prompts Library
This library implements the Taxonomy of User Needs and Actions (Authors: Renée Shelby, Fernando Diaz, Vinodkumar Prabhakaran,2025) from 1193 WildChat and ShareGPT logs to guide user simulator behavior.[arXiv:2510.06124 [cs.HC]]


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