# Evaluation Protocol & the Lead-Time Metric

## Why not accuracy
Onset windows are rare and the research question is about *earliness*. Plain accuracy is
dominated by the healthy majority and says nothing about how early we warn. We therefore
use ranking metrics for model selection and a domain **lead-time** metric for the claim.

## Primary metric — detection lead time
Implemented in `poultryai/eval/leadtime.py`.

Per flock, per disease:
1. Debounced alarm: fires at the first time the score stays >= threshold for
   `persistence` consecutive windows (default 3) — suppresses single-window spikes.
2. `lead_time = onset_time - alarm_time`  (positive = early; <=0 = detected but late).
3. Healthy flock with any alarm => a flock-level false positive.

Operating point: on the **validation** flocks, pick the most sensitive threshold whose
flock-level false-alarm rate <= target (default 5%) via `tune_threshold_for_fpr`, then
**freeze** it for the test flocks. Report:
- median lead time (hours),
- % of diseased flocks detected before onset,
- flock-level false-positive rate,
- lead time vs. the human-supervisor baseline (paired, per flock).

## Secondary metrics (`poultryai/train/metrics.py`)
- **Macro average-precision (PR-AUC)** — primary model-selection metric; robust to
  imbalance.
- **Macro-F1** at the tuned threshold.
- **Expected Calibration Error (ECE)** — 15-bin; operators act on the probability.

## Ablations (to isolate scientific contribution)
- Unimodal (env-only, audio-only, vision-only) vs. full fusion.
- Env backbone: TCN vs. PatchTST.
- With/without modality dropout (robustness to sensor loss).
- Audio on/off specifically for Infectious Bronchitis and Colibacillosis (the
  respiratory/air-quality-linked diseases).

## Statistical reporting
Report mean +/- std over >=5 seeds; paired test (Wilcoxon signed-rank) for lead-time vs.
baseline across flocks; 95% CIs via flock-level bootstrap.
