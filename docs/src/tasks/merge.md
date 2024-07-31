# 5. MERGE

The merging task will take the outputs of the aggregation task, and put the probabilities from each classifier into the light curve fit results (FITRES files) using SNID.

```yaml
MERGE:
  label:
    MASK: mask  # partial match on all sim, fit and agg
    MASK_SIM: mask  # partial match on sim
    MASK_FIT: mask  # partial match on lcfit
    MASK_AGG: mask  # partial match on aggregation task
```
