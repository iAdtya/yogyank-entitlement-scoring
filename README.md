## Timing (IST)
- **Start:** 2026-06-17, 15:30
- **End:** 2026-06-17, ~17:00
- **Total:** ~90 minutes

## Setup & run
```bash
pip install -r requirements.txt
python fixed_yogyank_training.py
```
## Output
<img width="621" height="128" alt="image" src="https://github.com/user-attachments/assets/0ebd204d-814c-4bb6-8111-8fce25f16102" />


## Files generated (in `artifacts/`, created when you run the script)
- `model_pipeline.pkl` — full pipeline (impute + one-hot + XGBoost).

## Key assumptions (feature availability)
- **Dropped (leakage):** `defaulted_in_next_12_months` — a future outcome.
- **Included** (assumed known before scoring): land area, historical repayment,
  liability ratio, annual income, rainfall deviation, NDVI, crop type, district,
  irrigation type, land ownership, soil type, sales channel.
- **Timing-sensitive** (would leak if measured after scoring): liability ratio,
  rainfall deviation, NDVI.
- **Model/policy separation:** `pm_kisan_status` is not a model feature; the
  PM-Kisan −150 rule is applied separately, after prediction.

## Completed vs skipped
- **Completed:** leakage removal, time-based validation, one-hot encoding,
  missing-value imputation, feature expansion + availability reasoning,
  model/policy separation, saved preprocessing `Pipeline`, reproducible
  artifacts (feature list, metrics, importances, metadata), reason codes,
  audit memo.
- **Skipped (time-box):** rolling time-series cross-validation and
  hyperparameter tuning (tuning deliberately skipped — the brief values
  judgement over leaderboard accuracy). See `audit_memo.md` §3.

## Validation approach
**Time-based split:** train on application years < 2024, test on 2024 — this
simulates scoring a future, unseen cohort (the random split in the draft leaks
future-period rows). Holdout **R² = 0.6957** — comparable to the draft's
0.6886.

**Do I trust it?** Directionally yes — the methodology is sound and leakage is
controlled.
