# LLM_NOTES — AI Tool Disclosure

## Tools used
- **Claude Code.** Used as a guide to explain each ML
  risk, propose fixes, and draft the documentation. I reviewed and committed
  every change myself, issue by issue.

## Where I used them
- Understanding the assessment and auditing `broken_yogyank_training.py`.
- Explaining each issue (leakage, validation, encoding, missing values,
  model/policy separation) and the fix for each.
- Drafting `audit_memo.md`, `README.md`, and this file.

## 3 actual prompts I used
1. "help understand the problem statement and also what should be my repo name …
   i hope you can read the files in yogyank folder"

## Suggestions I accepted
- The 7 fixes: drop leaky feature, time-based split, one-hot encoding, train-only
  median imputation, expanding to pre-scoring features, removing policy from the
  target, and applying PM-Kisan separately via `apply_policy()`.

## Suggestions I rejected / corrected
- **Verbose comments.** The assistant's first comment for the leakage fix was
  "future outcome, unknown at scoring time…"; I rejected it as over-explaining
  and shortened it to *"unavailable for new farmers (target leakage)"*.

## What I personally verified
- Ran `broken_yogyank_training.py` → confirmed `R² = 0.6886`.
- Confirmed the leaky feature, policy line, and `LabelEncoder` issues by reading
  the code; verified each fix in `fixed_yogyank_training.py`.
- Confirmed the time split partitions by year and that imputation uses the train
  median only.
- Reviewed and committed every change with its own message and reasoning.
