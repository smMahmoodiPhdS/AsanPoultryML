# Data Dictionary — on-disk contract

The training pipeline consumes `poultryai.data.schema.Window` objects. A concrete loader
(`scripts/train.py:load_windows`) must yield these from your storage. Recommended layout:

```
data/
  {owner}/{saloon}/{flock_id}/
    env.parquet         # columns: ts(epoch,s), temp_c, humidity_pct, lux, nh3_ppm, co2_ppm
    audio/{ts}.wav      # mono, >=16 kHz clips aligned to window ends
    vision.parquet      # columns: ts, activity_index, distribution_uniformity,
                        #          piling_score, feed_visit_rate, distress_index
    meta.json           # age_days, stocking_density, breed, mortality_rate_24h
    labels.parquet      # columns: ts, disease, onset_ts (veterinary-confirmed)
```

Canonical channel orders live in `schema.ENV_CHANNELS` / `schema.VISION_CHANNELS` — never
reorder; checkpoints depend on them.

## Windowing
- Env window: 240 min @ 1 sample/min (config `data.window_minutes`).
- Audio: last `data.audio_seconds` (default 4 s) ending at the window end.
- Vision: behavioural indices over the window (variable length; pooled in the model).
- `onset_offset[d] = onset_ts[d] - window_end` (seconds); +inf if the flock never gets
  disease d. This drives the lead-time evaluation and onset-aware labelling.

## Labelling policy
- A window is positive for disease d if it lies within a pre-onset horizon H before
  `onset_ts[d]` (e.g. H = 72 h) OR within the clinical period. H is a studied
  hyperparameter (too small => nothing to learn early; too large => label noise).
- Ground truth is veterinary diagnosis + mortality/performance records.
