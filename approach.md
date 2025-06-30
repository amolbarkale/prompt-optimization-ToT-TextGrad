          ┌─────────────────────┐
          │  Domain Tasks (5-7) │
          └────────┬────────────┘
                   ↓
       ┌───────────────────────────────┐
       │ Tree-of-Thought Reasoning (ToT)│
       └────────┬──────────────┬───────┘
                ↓              ↓
       ┌────────────┐   ┌─────────────┐
       │ Self-Consistency │ → Aggregate Answers
       └──────┬─────┘
              ↓
 ┌────────────────────────────┐
 │ Prompt Optimization Loop   │ ←─────┐
 │ (TextGrad / OPRO-style)    │       │
 └────┬───────────────────────┘       │
      ↓                               │
  Update Prompts → Rerun → Evaluate ──┘
