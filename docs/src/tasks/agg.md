# 4. AGGREGATION

The aggregation task takes results from one or more classification tasks (that have been run in predict mode on the same dataset) and generates comparisons between the classifiers (their correlations, PR curves, ROC curves and their calibration plots). Additionally, it merges the results of the classifiers into a single csv file, mapping SNID to one column per classifier.

```yaml
AGGREGATION:
  SOMELABEL:
    MASK: mask  # Match sim AND classifier
    MASK_SIM: mask # Match only sim
    MASK_CLAS: mask # Match only classifier
    RECALIBRATION: SIMNAME # Optional, use this simulation to recalibrate probabilities. Default no recal.
    # Optional, changes the probability column name of each classification task listed into the given probability column name.
    # Note that this will crash if the same classification task is given multiple probability column names.
    # Mostly used when you have multiple photometrically classified samples
    MERGE_CLASSIFIERS:
      PROB_COLUMN_NAME: [CLASS_TASK_1, CLASS_TASK_2, ...]
    OPTS:
      PLOT: True # Default True, make plots
      PLOT_ALL: False # Default False. Ie if RANSEED_CHANGE gives you 100 sims, make 100 set of plots.
```
