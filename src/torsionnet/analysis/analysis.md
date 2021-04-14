# Logging and Analysis Module

## Tensorboard logging features
- Reward (Gibbs Score) for each training episode (across all episdoes).
- Reward for each evaluation episode (across all episodes).
- Run time for each training step.
- Run time for each model update.
- Overall runtime and wall time.
- Training loss for each timestep.


## Jupyter logging/analysis features
- View results from **single** evaluation episodes and compare across different agents/setups.
- Reward (Gibbs score) for each evaluation episode.
- Energy of every conformer generated during the evaluation episode (as seen in histogram).
- Reward for every conformer generated.
- Number of duplicate molecules generated.
- TFD of all the molecules generated.
- Diversitiy metric (based on TFD).
- Molecule viewer.
- Saved .mol files.