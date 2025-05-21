#!/usr/bin/env python3
"""
plot.py

Module to plot the convergence of qNN training methods.

Authors:
- Pedro C. Delbem <pedrodelbem@usp.br>
"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import re

# Ask the user to input the folder path
folder_path = "results/"+input("Enter the path to the folder containing the CSV files: ")
folder = Path(folder_path)

# Create the "plots" folder if it doesn't exist
plots_folder = Path("plots")
plots_folder.mkdir(exist_ok=True)

# Collect all CSV files
csv_files = list(folder.glob("*.csv"))
if not csv_files:
    print("No .csv files found in the specified folder.")
    exit()

# Load and normalize all files
all_data = []
gate_to_tolerance = {}

for file in csv_files:
    try:
        df = pd.read_csv(file, sep=";")
        df.columns = df.columns.str.strip().str.lower()
        if df.empty:
            print(f"Skipping empty file: {file.name}")
            continue

        required = {"logic gate", "method", "iteration", "error", "run"}
        if not required.issubset(df.columns):
            print(f"Missing columns in {file.name}: {df.columns}")
            continue

        df["iteration"] = df["iteration"].astype(int)
        df["error"] = df["error"].astype(float)
        df["run"] = df["run"].astype(int)

        logic_gate = df["logic gate"].iloc[0]

        # Only extract tolerance once per logic gate
        if logic_gate not in gate_to_tolerance:
            match = re.search(r'tolerance[_\-]?([0-9.]+)', file.stem)
            if match:
                tolerance = float(match.group(1))
                gate_to_tolerance[logic_gate] = tolerance
            else:
                print(f"Warning: Couldn't extract tolerance from {file.name}. Using 0.")
                gate_to_tolerance[logic_gate] = 0.0

        all_data.append(df)

    except Exception as e:
        print(f"Error with file {file.name}: {e}")

if not all_data:
    print("No valid data found.")
    exit()

# Merge all DataFrames
full_df = pd.concat(all_data, ignore_index=True)

# Extract the name of the parent directory (for file naming)
parent_directory_name = folder.name

# Create one interactive plot per logic gate
for logic_gate in full_df["logic gate"].unique():
    df_gate = full_df[full_df["logic gate"] == logic_gate]
    fig = go.Figure()

    for method in df_gate["method"].unique():
        df_method = df_gate[df_gate["method"] == method]

        # Plot individual runs (transparent)
        for run in df_method["run"].unique():
            df_run = df_method[df_method["run"] == run]
            fig.add_trace(go.Scatter(
                x=df_run["iteration"],
                y=df_run["error"],
                mode="lines+markers",
                name=f"{method} (run {run})",
                line=dict(width=1),
                opacity=0.3,
                hoverinfo="x+y+name",
                legendgroup=method,
                showlegend=False
            ))

        # Group by iteration and compute summary statistics
        summary = df_method.groupby("iteration")["error"].agg(["mean", "std", "min", "max"]).reset_index()

        # Shaded area between min and max
        fig.add_trace(go.Scatter(
            x=summary["iteration"],
            y=summary["max"],
            mode="lines",
            line=dict(width=0),
            name=f"{method} (max)",
            hoverinfo="skip",
            legendgroup=method,
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=summary["iteration"],
            y=summary["min"],
            mode="lines",
            fill='tonexty',
            fillcolor='rgba(0,100,200,0.1)',
            line=dict(width=0),
            name=f"{method} min/max range",
            hoverinfo="skip",
            legendgroup=method,
            showlegend=False
        ))

        # Mean with standard deviation bars
        fig.add_trace(go.Scatter(
            x=summary["iteration"],
            y=summary["mean"],
            mode="markers+lines",
            name=f"{method} (mean)",
            line=dict(width=2),
            marker=dict(size=6),
            error_y=dict(
                type='data',
                symmetric=False,
                array=summary["std"],
                arrayminus=summary["std"],
                visible=True
            ),
            legendgroup=method
        ))

    # Add tolerance line (once per logic gate)
    tolerance = gate_to_tolerance.get(logic_gate, None)
    if tolerance is not None:
        max_iter = df_gate["iteration"].max()
        fig.add_trace(go.Scatter(
            x=[0, max_iter],
            y=[tolerance, tolerance],
            mode="lines",
            name=f"Tolerance = {tolerance}",
            line=dict(color="black", dash="dash", width=1.5),
            hoverinfo="name+y"
        ))

    fig.update_layout(
        title=f"Interactive Convergence Plot — {logic_gate.upper()}",
        xaxis_title="Iteration",
        yaxis_title="Error",
        hovermode="closest",
        template="plotly_white",
        legend_title="Methods",
        height=700
    )

    # Define file names with parent directory name and logic gate
    output_html = plots_folder / f"{parent_directory_name}_interactive_plot_{logic_gate}.html"
    output_png = plots_folder / f"{parent_directory_name}_plot_{logic_gate}.png"

    # Save interactive plot as HTML
    fig.write_html(output_html)
    print(f"Saved interactive plot: {output_html}")

    # Save static plot as PNG
    fig.write_image(output_png)
    print(f"Saved static plot: {output_png}")
