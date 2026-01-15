# curlers
CSAS 2026 Data Challenge – Mixed Doubles curling analysis.

This repo builds an end-level expected points (EP) model and uses a dynamic
programming (MDP) policy to analyze optimal Power Play (PP) timing. It also
produces diagnostics and evaluation plots, plus processed CSVs for analysis.

## Project layout
- `code/`: main pipeline + helper modules
- `data/raw/`: raw competition CSVs (Stones, Ends, Games, Teams, etc.)
- `data/processed/`: exported datasets used for analysis
- `results/`: generated plots and figures

## Quick start
Run the full pipeline (data prep → EDA → modeling → policy → evaluation):

```bash
python3 code/main.py
```

## Outputs
Running `code/main.py` generates:

### Processed data
- `data/processed/end_level_start.csv`  
  One row per team per end at start-of-end state (ref = team under observation).
- `data/processed/pp_decision_points.csv`  
  Subset of decision-eligible states (hammer + PP available, End 1–8).
- `data/processed/pp_decision_evaluation.csv`  
  Actual vs optimal decisions with win-probability differences.
- `data/processed/pp_team_stats.csv`  
  Aggregated team-level decision statistics.

### Plots
- `results/ep/`  
  EP model diagnostics: `confusion_matrix.png`, `distribution.png`,
  `feature_importance.png`.
- `results/power-play/`  
  PP policy heatmaps, decision quality, accuracy heatmaps, and team summaries.
  Notable files include:
  - `pp_heatmap_opp_saved.png` (opponent PP still available)
  - `pp_heatmap_opp_used.png` (opponent PP already used)
  - `pp_decision_heatmap.png`
  - `pp_accuracy_heatmap.png`
  - `pp_team_performance.png`, `pp_team_accuracy.png`

### EDA plots
EDA figures are saved to `results/eda/` (e.g., usage by end and score diff).
