# edtrack-fatigue-analysis

Analyzing student learning patterns in the Open University Learning Analytics Dataset (OULAD): when online learners disengage, what predicts it, and what interface interventions might help.

**Start here: [`case_study.ipynb`](case_study.ipynb)**, a readable walk-through of the whole project with the outputs saved, so it renders on GitHub without running anything.

## What's in the case study

1. The OULAD data and the questions asked
2. Cleaning and merging the seven tables
3. Defining engagement over time (active days as sessions)
4. Engagement vs. scores: how often students show up beats how long they stay
5. The drop-off point: early exit (withdrawers, ~week 7) vs. slow fade (fails, ~week 14)
6. The early-warning signal: zero active days in weeks 0–3 means a 5% pass rate
7. Two personas, Jordan and Sam, explained with cognitive load and spacing
8. Three interventions: C2 "One small thing", C6 "Deadline runway", C7 "Welcome back"
9. A simulated before/after of those interventions (**illustrative only, not evidence**)
10. Limitations and next steps

## Repo layout

| Path | What it is |
|---|---|
| `case_study.ipynb` | The case study notebook |
| `oulad_clean.py` | Loads, cleans and merges the OULAD tables into one row per enrolment |
| `engagement_over_time.py` | Weekly engagement panel, pre-deadline windows, drop-off stats and charts |
| `simulate_intervention.py` | Illustrative before/after simulation of C2, C6 and C7 |
| `figures/` | Charts from the real-data analysis |
| `outputs/simulated/` | Simulation outputs (illustrative) |
| `data/` | Put the raw OULAD CSVs here (not committed, see `data/README.md`) |

## Running it

```bash
pip install -r requirements.txt
# put the OULAD CSVs in data/, or: export OULAD_DATA_DIR=/path/to/oulad
jupyter lab case_study.ipynb
```

The scripts also run on their own: `python engagement_over_time.py` writes `processed/` tables and `figures/`, and `python simulate_intervention.py` rewrites `outputs/simulated/`.
