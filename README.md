# GRAD 505 — Foundations in Data Science
**Purdue University | Summer 2026**
**Author:** Bradley May

---

## Project Overview

This repository contains all Python code and generated visualizations for the GRAD 505 course project. The dataset used is the **MLB Pitch Data (2015–2018)**, a pitch-level record of every Major League Baseball regular-season game across four seasons (~2.87 million pitches).

| Script | Deliverable |
|---|---|
| `dimensions.py` | Project Deliverable 01 |
| `visualizations.py` | Project Deliverable 02 |

### Research Questions
1. Does a pitcher's pitch-type distribution shift when facing the same batter for the 2nd or 3rd time in a single game?
2. To what extent does the spin rate of a Four-Seam Fastball predict the probability of a swing and miss (whiff)?
3. Is there a statistically significant correlation between a pitcher's average fastball velocity and their zone percentage during the high-leverage late innings (7th–9th)?

---

## Repository Structure

```
GRAD_505_PROJECT/
├── dimensions.py           # Project Deliverable 01 — dataset shape, dtypes, and null counts
├── visualizations.py       # Project Deliverable 02 — research question charts
├── DATASET/                # CSV data files (not tracked by git — see below)
│   ├── pitches.csv
│   ├── atbats.csv
│   ├── ejections.csv
│   ├── games.csv
│   └── player_names.csv
└── IMAGES/                 # PNG outputs from visualizations.py
```

---

## Setup

### 1. Download the Dataset

The CSV files in `DATASET/` are excluded from this repository because `pitches.csv` exceeds 1 GB. Download the full dataset from Kaggle and place all five CSV files in the `DATASET/` folder:

**Dataset:** [MLB Pitch Data (2015–2018)](https://www.kaggle.com/datasets/pschale/mlb-pitch-data-20152018)

Files needed:
- `pitches.csv`
- `atbats.csv`
- `ejections.csv`
- `games.csv`
- `player_names.csv`

### 2. Install Dependencies

```bash
pip install pandas matplotlib seaborn numpy
```

### 3. Run the Scripts

```bash
# Project Deliverable 01 — report dataset dimensions and data quality
python dimensions.py

# Project Deliverable 02 — generate all visualizations (saved to IMAGES/)
python visualizations.py
```

---

## Output

Running `visualizations.py` (Project Deliverable 02) produces six PNG files in `IMAGES/`:

| File | Description |
|---|---|
| `fig1_missing_data.png` | Missing values by column |
| `fig2_q1_pitch_type_tto.png` | Q1 — pitch-type distribution by times through order |
| `fig3_q1_table.png` | Q1 — pitch-type share summary table |
| `fig4_q2_spinrate_boxplot.png` | Q2 — spin rate distribution, whiff vs. contact |
| `fig5_q2_spinrate_whiff.png` | Q2 — spin rate bins vs. whiff rate |
| `fig6_q3_velo_zone.png` | Q3 — avg fastball velocity vs. zone % (innings 7–9) |
