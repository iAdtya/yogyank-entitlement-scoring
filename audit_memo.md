# Audit Memo — Yogyank Entitlement Score Baseline

Yogyank outputs a bank-agnostic **Entitlement Score**

## 1. What was dangerous in the original script

1. **Target leakage.** `defaulted_in_next_12_months` was used as a feature. It
   is a *future* outcome, unknown for a new farmer at scoring time, so the model
   learned from information it will never have in production.
2. **Policy baked into the target.** `target_entitlement_score -= 150` for
   PM-Kisan = "No" welded a business rule into the label, so the model was
   forced to learn a policy. This hides the rule and corrupts the metric.
3. **Random split.** `shuffle=True` mixed all years (2022–2024) into train and
   test. For a system meant to score *future* farmers, this lets the model peek
   at the same period it is tested on.
4. **`LabelEncoder` on features.** It imposed a fake order on nominal categories
   (e.g. Rice=2 > Cotton=0), was reused across columns, and had no handling for
   unseen categories.
5. **No missing-value handling** for `rainfall_deviation_pct` / `ndvi_score`
   (~15% null each).
6. **No reproducibility or explainability** — only a raw `.pkl`, no feature
   list, schema, metrics record, or reason codes.

**Why the reported R² was not trustworthy:** the draft printed
`R² = 0.6886 (Wow!)`, but it was produced by an invalid procedure — a leaky
feature (1) under a non-temporal split (3) on a policy-polluted label (2). The
number cannot represent future-scoring performance regardless of its value.

For comparison, the rebuilt baseline scores **R² = 0.6957** on a genuine 2024
future holdout with no leakage — essentially the same number, but now honest and
defensible.

## 2. What I changed

| Issue | Change | Why it's better |
|---|---|---|
| Leakage | dropped `defaulted_in_next_12_months` | removes info unavailable at scoring |
| Validation | time split (train < 2024, test = 2024) | simulates scoring a future cohort |
| Policy | removed from target; new `apply_policy()` step | model/policy separated, rule auditable & changeable without retraining |
| Encoding | `LabelEncoder` → `pd.get_dummies` (one-hot) | no fake ordering on categories |
| Missing values | impute with **train** median, applied to both | explicit, no test→train leakage |
| Features | expanded to all pre-scoring features; documented exclusions | uses real signal, states availability assumptions |
| Reproducibility | (artifacts: feature list, metrics, metadata) | re-runnable and reviewable |

**Feature-availability assumptions** (definitions were intentionally withheld):
included only features judged known at application time; flagged
`liability_ratio_pct`, `rainfall_deviation_pct`, `ndvi_score` as
timing-sensitive (would leak if measured *after* scoring); excluded
`defaulted_in_next_12_months`, `farmer_id`, `application_year`.

## 3. Limitations remaining

- **Would not trust yet:** the exact accuracy and the timing assumptions on
  liability/rainfall/NDVI — the split rests on only 3 years with a single-year
  holdout, so R² is a direction, not a guarantee.
- **Would improve with more time:** wrap preprocessing in a saved `Pipeline`,
  use rolling time-series cross-validation.
