"""
visualizations.py
-----------------
Project Deliverable 02 — generate all research-question visualizations
for GRAD 505.

Outputs 6 PNG files to IMAGES/:
  fig1_missing_data.png        - Missing values by column
  fig2_q1_pitch_type_tto.png   - Q1: pitch-type distribution by times through order
  fig3_q1_table.png            - Q1: summary table (pitch-type % by TTO)
  fig4_q2_spinrate_boxplot.png - Q2: spin rate distribution, whiff vs. contact
  fig5_q2_spinrate_whiff.png   - Q2: spin rate bins vs. whiff rate (scatter)
  fig6_q3_velo_zone.png        - Q3: avg fastball velocity vs. zone % (late innings)

Research questions:
  Q1 - Does a pitcher's pitch-type distribution shift when facing the same
       batter for the 2nd or 3rd time in a single game?
  Q2 - To what extent does the spin rate of a Four-Seam Fastball predict the
       probability of a swing and miss (whiff)?
  Q3 - Is there a statistically significant correlation between a pitcher's
       average fastball velocity and their zone percentage during innings 7-9?

Run this script from the GRAD_505_PROJECT folder.
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


# Build file paths relative to where this script is saved, so the script works regardless of which directory you run it from.
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(SCRIPT_DIR, "DATASET")
IMAGES_FOLDER  = os.path.join(SCRIPT_DIR, "IMAGES")

# Create the IMAGES folder if it does not already exist
os.makedirs(IMAGES_FOLDER, exist_ok=True)


# Chart style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})


# Load data 
print("Loading data...")
pitches = pd.read_csv(os.path.join(DATASET_FOLDER, "pitches.csv"))
atbats  = pd.read_csv(os.path.join(DATASET_FOLDER, "atbats.csv"))
print(f"  pitches : {len(pitches):,} rows")
print(f"  at-bats : {len(atbats):,} rows")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Missing-value audit (key columns in pitches.csv)
# ════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: missing data...")

# Columns most relevant to the research questions
KEY_COLS = [
    "px", "pz", "start_speed", "end_speed", "spin_rate", "break_angle",
    "break_length", "pfx_x", "pfx_z", "zone", "code", "type", "pitch_type",
    "ab_id", "b_count", "s_count", "outs", "on_1b", "on_2b", "on_3b",
]

# Calculate what percentage of each column is missing
null_pct        = pitches[KEY_COLS].isna().mean() * 100
null_df         = null_pct.reset_index()
null_df.columns = ["Column", "Null %"]
null_df         = null_df.sort_values("Null %", ascending=True)

# Assign a color to each bar: red if missing values exist, green if none
colors = []
for value in null_df["Null %"]:
    if value > 0:
        colors.append("#c0392b")  # red — has missing values
    else:
        colors.append("#27ae60")  # green — fully complete

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(null_df["Column"], null_df["Null %"], color=colors)
ax.set_xlabel("Missing Values (%)")
ax.set_title("Figure 1 — Missing Data by Column (pitches.csv, key fields)")
ax.xaxis.set_major_formatter(mtick.PercentFormatter())

# Add a percentage label to the end of each bar that has missing values
for bar, val in zip(bars, null_df["Null %"]):
    if val > 0:
        # Position the label just to the right of the bar, vertically centered
        label_x = val + 0.02
        label_y = bar.get_y() + bar.get_height() / 2
        ax.text(label_x, label_y, f"{val:.2f}%", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_FOLDER, "fig1_missing_data.png"), bbox_inches="tight")
plt.close()
print("  saved fig1_missing_data.png")

# ════════════════════════════════════════════════════════════════════════════
# Research Q1 SETUP — engineer the Times Through the Order (TTO) variable
# ════════════════════════════════════════════════════════════════════════════
print("\nBuilding Times Through the Order variable for Q1...")

# Keep only the columns needed from each table
pitch_types = pitches[["ab_id", "pitch_type"]].dropna(subset=["pitch_type"])
at_bat_info = atbats[["ab_id", "batter_id", "pitcher_id", "g_id"]].copy()

# Sort at-bats by ab_id (chronological order within each game)
at_bat_info = at_bat_info.drop_duplicates("ab_id").sort_values("ab_id")

# Count how many times each batter has faced each pitcher in each game.
# cumcount() numbers each row within a group starting at 0, so we add 1.
at_bat_info["tto"] = (
    at_bat_info.groupby(["g_id", "pitcher_id", "batter_id"]).cumcount() + 1
)


# Convert the numeric TTO count to a readable label using a plain function
def assign_tto_label(tto_count):
    """Return '1st', '2nd', or '3rd+' based on the times-through-order count."""
    if tto_count == 1:
        return "1st"
    elif tto_count == 2:
        return "2nd"
    else:
        return "3rd+"

at_bat_info["tto_label"] = at_bat_info["tto"].apply(assign_tto_label)
print(f"  TTO counts: {at_bat_info['tto_label'].value_counts().to_dict()}")

# Join pitch types onto the at-bat TTO labels
merged = pitch_types.merge(at_bat_info[["ab_id", "tto_label"]], on="ab_id")

# Work with only the 6 most common pitch types to keep the chart readable
top_pitch_types = merged["pitch_type"].value_counts().head(6).index
merged_top      = merged[merged["pitch_type"].isin(top_pitch_types)]

# Count pitches by TTO group and pitch type
tto_counts = (
    merged_top
    .groupby(["tto_label", "pitch_type"])
    .size()
    .reset_index(name="count")
)

# Calculate total pitches per TTO group, then map that total back onto each row
group_totals          = tto_counts.groupby("tto_label")["count"].sum()
tto_counts["total"]   = tto_counts["tto_label"].map(group_totals)
tto_counts["pct"]     = (tto_counts["count"] / tto_counts["total"]) * 100

# Reshape so that rows are TTO labels and columns are pitch types
tto_pivot = tto_counts.pivot(index="tto_label", columns="pitch_type", values="pct")
tto_pivot = tto_pivot.reindex(["1st", "2nd", "3rd+"])  # ensure correct row order

print("  Pitch-type distribution by TTO (%):")
print(tto_pivot.round(2).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2, for Research Q1: Grouped bar chart — pitch-type % by TTO
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 2: Q1 grouped bar chart...")

fig, ax = plt.subplots(figsize=(10, 6))
tto_pivot.plot(kind="bar", ax=ax, width=0.75, edgecolor="white")
ax.set_xlabel("Times Through the Order")
ax.set_ylabel("Share of Pitches (%)")
ax.set_title("Figure 2 — Pitch-Type Distribution by Times Through the Order (Q1)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(title="Pitch Type", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.set_xticklabels(["1st", "2nd", "3rd+"], rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_FOLDER, "fig2_q1_pitch_type_tto.png"), bbox_inches="tight")
plt.close()
print("  saved fig2_q1_pitch_type_tto.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3, for Research Q1: Summary table image
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 3: Q1 summary table...")

tbl             = tto_pivot.round(2).reset_index()
tbl.columns.name = None

# Build the table data row by row
table_data = []
for _, row in tbl.iterrows():
    row_data = [row["tto_label"]]
    for col in tto_pivot.columns:
        row_data.append(f"{row[col]:.2f}%")
    table_data.append(row_data)

col_labels = ["Times Through Order"] + list(tto_pivot.columns)

fig, ax = plt.subplots(figsize=(9, 3))
ax.axis("off")  # hide the axes — we only want the table
table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.6)

# Style the header row and alternate data rows for readability
for (row_idx, col_idx), cell in table.get_celld().items():
    if row_idx == 0:
        cell.set_facecolor("#2E4057")
        cell.set_text_props(color="white", weight="bold")
    elif row_idx % 2 == 0:
        cell.set_facecolor("#f0f4f8")  # light blue for every other row

ax.set_title(
    "Table 1 — Pitch-Type Share (%) by Times Through the Order (Q1)",
    fontweight="bold", pad=12,
)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_FOLDER, "fig3_q1_table.png"), bbox_inches="tight")
plt.close()
print("  saved fig3_q1_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4, for Research Q2: Box plot — spin rate, whiff vs. contact (Four-Seam Fastball)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 4: Q2 spin rate box plot...")

# Filter to Four-Seam Fastballs only, dropping rows with missing spin or code
ff = pitches[pitches["pitch_type"] == "FF"].copy()
ff = ff[ff["spin_rate"].notna()]
ff = ff[ff["code"].notna()]


# Label each pitch as a whiff or not using a plain function
def get_pitch_result(code):
    """Return 'Whiff (S)' for swinging strikes, 'Contact / Other' for everything else."""
    if code == "S":
        return "Whiff (S)"
    else:
        return "Contact / Other"

ff["result"] = ff["code"].apply(get_pitch_result)

fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(
    data=ff,
    x="result",
    y="spin_rate",
    palette=["#e74c3c", "#3498db"],
    width=0.5,
    flierprops=dict(marker=".", alpha=0.2, markersize=2),  # style the outlier dots
    ax=ax,
)
ax.set_xlabel("")
ax.set_ylabel("Spin Rate (RPM)")
ax.set_title("Figure 3 — Four-Seam Fastball Spin Rate: Whiff vs. Contact (Q2)")

# Add a median label above each box
medians = ff.groupby("result")["spin_rate"].median()
for i, (label, median_value) in enumerate(medians.items()):
    ax.text(i, median_value + 30, f"Median: {median_value:,.0f}",
            ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_FOLDER, "fig4_q2_spinrate_boxplot.png"), bbox_inches="tight")
plt.close()
n_whiff   = (ff["result"] == "Whiff (S)").sum()
n_contact = (ff["result"] == "Contact / Other").sum()
print(f"  saved fig4_q2_spinrate_boxplot.png  (whiff={n_whiff:,}  contact={n_contact:,})")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5, for Research Q2: Scatter — spin rate bins vs. whiff rate
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 5: Q2 spin rate vs. whiff rate scatter...")

# Create a binary whiff column (1 = swinging strike, 0 = everything else)
ff["whiff"] = (ff["code"] == "S").astype(int)

# Divide spin rates into 20 equal-width bins
ff["spin_bin"] = pd.cut(ff["spin_rate"], bins=20)

# For each bin, calculate the whiff rate and total pitch count
bin_stats = ff.groupby("spin_bin", observed=True)["whiff"].agg(["mean", "count"])
bin_stats = bin_stats.reset_index()
bin_stats.columns = ["spin_bin", "whiff_rate", "count"]

# Get the midpoint of each bin interval for use as the x-axis value
bin_stats["spin_mid"] = bin_stats["spin_bin"].apply(lambda interval: interval.mid)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each bin as a bubble — larger bubbles = more pitches in that bin
ax.scatter(
    bin_stats["spin_mid"],
    bin_stats["whiff_rate"] * 100,
    s=bin_stats["count"] / 50,   # scale bubble size by pitch count
    alpha=0.75,
    color="#8e44ad",
    edgecolors="white",
    linewidth=0.5,
)

# Fit and draw a linear trend line using numpy
coefficients = np.polyfit(bin_stats["spin_mid"], bin_stats["whiff_rate"] * 100, deg=1)
trend_line   = np.poly1d(coefficients)
x_values     = np.linspace(bin_stats["spin_mid"].min(), bin_stats["spin_mid"].max(), 100)
ax.plot(x_values, trend_line(x_values), "r--", linewidth=1.5, label="Trend line")

ax.set_xlabel("Spin Rate (RPM)")
ax.set_ylabel("Whiff Rate (%)")
ax.set_title(
    "Figure 4 — Four-Seam Fastball Spin Rate vs. Whiff Rate (Q2)\n"
    "(bubble size = pitch count in bin)"
)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_FOLDER, "fig5_q2_spinrate_whiff.png"), bbox_inches="tight")
plt.close()
print("  saved fig5_q2_spinrate_whiff.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6, for Research Q3: Scatter — avg fastball velocity vs. zone % in innings 7–9
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 6: Q3 velocity vs. zone% scatter (late innings)...")

FASTBALL_TYPES = ["FF", "FT", "FC"]
MIN_PITCHES    = 50  # only include pitchers with at least this many late-inning fastballs

# Filter pitches to fastball types and drop rows missing speed or zone data
fastballs = pitches[pitches["pitch_type"].isin(FASTBALL_TYPES)].copy()
fastballs = fastballs[["ab_id", "start_speed", "zone", "pitch_type"]].dropna()

# Join in the inning column from at-bats
fastballs = fastballs.merge(atbats[["ab_id", "pitcher_id", "inning"]], on="ab_id")

# Keep only late-inning pitches (7th inning onward)
late_innings = fastballs[fastballs["inning"] >= 7].copy()

# Create a binary in-zone column: 1 if the pitch was in zones 1–9, 0 otherwise
late_innings["in_zone"] = late_innings["zone"].between(1, 9).astype(int)

# Aggregate to one row per pitcher: average velocity and zone percentage
pitcher_stats = late_innings.groupby("pitcher_id").agg(
    avg_velo=("start_speed", "mean"),
    zone_pct=("in_zone", "mean"),
    n=("start_speed", "count")
).reset_index()

# Remove pitchers with too few pitches for a reliable estimate
pitcher_stats = pitcher_stats[pitcher_stats["n"] >= MIN_PITCHES]
pitcher_stats["zone_pct"] = pitcher_stats["zone_pct"] * 100  # convert to percentage

# Calculate the Pearson correlation coefficient between velocity and zone%
corr = pitcher_stats["avg_velo"].corr(pitcher_stats["zone_pct"])

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(
    pitcher_stats["avg_velo"],
    pitcher_stats["zone_pct"],
    alpha=0.45,
    s=30,
    color="#27ae60",
    edgecolors="none",
)

# Fit and draw a linear trend line
coefficients = np.polyfit(pitcher_stats["avg_velo"], pitcher_stats["zone_pct"], deg=1)
trend_line   = np.poly1d(coefficients)
x_values     = np.linspace(pitcher_stats["avg_velo"].min(), pitcher_stats["avg_velo"].max(), 100)
ax.plot(x_values, trend_line(x_values), "r--", linewidth=1.5,
        label=f"Trend (slope={coefficients[0]:.3f})")

ax.set_xlabel("Average Fastball Velocity (MPH)")
ax.set_ylabel("Zone Percentage (%)")
ax.set_title(
    f"Figure 5 — Avg Fastball Velocity vs. Zone % in Innings 7–9 (Q3)\n"
    f"Pearson r = {corr:.3f}  |  n = {len(pitcher_stats):,} pitchers"
)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_FOLDER, "fig6_q3_velo_zone.png"), bbox_inches="tight")
plt.close()
print(f"  saved fig6_q3_velo_zone.png  (r={corr:.4f}, n={len(pitcher_stats)} pitchers)")

print("\nAll figures saved to IMAGES/")
