# Models

The detector is rule-based (patient-specific baseline + robust threshold), so there is nothing to train or save.
The only fitted model is the Isolation-Forest *baseline* used for comparison inside `src/evaluate.py`.
