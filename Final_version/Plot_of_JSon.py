#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ============================================================
# ADD YOUR FILES HERE
# ============================================================
FILES = [
    "outputs/RUN1_SUM.json",
    "outputs/RUN2_SUM.json",
    "outputs/RUN3_SUM.json",
    "outputs/RUN4_SUM.json",
    "outputs/RUN5_SUM.json"

]

OUTPUT_FILE = "combined_results.csv"
TABLE_PLOT_FILE = "combined_results_table.png"


def plot_results_table(df, output_file):
    if df.empty:
        print("No results found; skipping table plot.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed; skipping table plot. "
            "Install with: pip install matplotlib"
        )
        return

    df_plot = df.copy()

    if "Accuracy" in df_plot.columns:
        acc = pd.to_numeric(df_plot["Accuracy"], errors="coerce")
        df_plot["Accuracy"] = acc.map(
            lambda x: "" if pd.isna(x) else f"{x:.4f}"
        )

    if "Count" in df_plot.columns:
        cnt = pd.to_numeric(df_plot["Count"], errors="coerce")
        df_plot["Count"] = cnt.map(
            lambda x: "" if pd.isna(x) else str(int(x))
        )

    df_plot = df_plot.fillna("").astype(str)

    n_rows, n_cols = df_plot.shape

    fig_width = max(8, n_cols * 1.8)
    fig_height = max(2.5, min(40, 0.45 * (n_rows + 1)))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df_plot.values,
        colLabels=df_plot.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)

    fig.tight_layout()
    output_path = Path(output_file)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved table plot to: {output_path}")


def parse_result_file(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for key, value in data.items():
        # Expected format:
        # "MODEL|DATASET|LLM"
        parts = key.split("|")

        if len(parts) != 3:
            print(f"Skipping invalid key in {path.name}: {key}")
            continue

        model, dataset, llm = parts

        rows.append({
            "Model": model,
            "Dataset": dataset,
            "LLM": llm.strip().replace("Qwen/", ""),  # Remove "Qwen/" prefix if present
            "Accuracy": value.get("accuracy"),
            "Count": value.get("count"),
            "Source File": path.name,
        })

    return rows


# ============================================================
# LOAD ALL FILES
# ============================================================
all_rows = []

for file in FILES:
    all_rows.extend(parse_result_file(file))


# ============================================================
# NUMBER DUPLICATE RUNS
# Same MODEL + LLM => Run 1, Run 2, ...
# ============================================================
run_counter = defaultdict(int)

for row in all_rows:
    key = (row["Model"], row["LLM"])

    run_counter[key] += 1
    run_number = run_counter[key]

    row["Run"] = f"Run {run_number}"


# ============================================================
# CREATE TABLE
# ============================================================
df = pd.DataFrame(all_rows)

# Reorder columns
df = df[
    [
        "Model",
        "Dataset",
        "LLM",
        "Run",
        "Accuracy",
        "Count",
        "Source File",
    ]
]

# Sort nicely
df = df.sort_values(
    by=["Model", "LLM", "Run"],
    kind="stable"
).reset_index(drop=True)


# ============================================================
# SAVE + PRINT
# ============================================================
df.to_csv(OUTPUT_FILE, index=False)

print("\n=== RESULTS TABLE ===\n")
print(df.to_string(index=False))

print(f"\nSaved CSV to: {OUTPUT_FILE}")

plot_results_table(df, TABLE_PLOT_FILE)