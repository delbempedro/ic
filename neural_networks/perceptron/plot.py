#!/usr/bin/env python3
"""
plot.py

Module to plot the convergence of qNN trainin methods.

Dependencies:
- Use os to handle file paths
- Use pandas to organize the data
- Use plotly.io to save the plots as PNG and HTML
- Use plotly.graph_objects to create the plots

Since:
- 04/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""
import os
import argparse
import pandas as pd # type: ignore
import plotly.graph_objects as go # type: ignore
import plotly.io as pio # type: ignore

def load_and_prepare_data(folder_paths, convergence_threshold=0.25):
    df_list = []
    for folder_path in folder_paths:
        all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        all_paths = [os.path.join(folder_path, f) for f in all_files]

        for path in all_paths:
            df = pd.read_csv(path, sep=";")
            df.columns = df.columns.str.strip().str.lower()  # Normalize column names
            df_list.append(df)

    full_df = pd.concat(df_list, ignore_index=True)

    # Convert relevant columns to appropriate types
    full_df["iteration"] = full_df["iteration"].astype(int)
    full_df["error"] = full_df["error"].astype(float)
    full_df["run"] = full_df["run"].astype(int)

    # Ensure all iterations are present per (logic gate, method, run)
    completed_runs = []
    global_max_iter = full_df["iteration"].max()

    for (gate, method, run), run_df in full_df.groupby(["logic gate", "method", "run"]):
        # Handle duplicates: average repeated iterations
        run_df = run_df.groupby("iteration").mean(numeric_only=True)

        # Reindex to ensure all iterations are present
        run_df = run_df.reindex(range(global_max_iter + 1))

        # Add additional columns for identification
        run_df["logic gate"] = gate
        run_df["method"] = method
        run_df["run"] = run

        completed_runs.append(run_df.reset_index())

    full_df = pd.concat(completed_runs, ignore_index=True)
    full_df.rename(columns={"index": "iteration"}, inplace=True)

    return full_df, global_max_iter

def generate_and_save_plots(full_df, output_root_folder, convergence_threshold, max_iteration):
    output_folder = os.path.join(output_root_folder, "plots")
    os.makedirs(output_folder, exist_ok=True)

    grouped_by_logic = full_df.groupby("logic gate")

    for logic_gate, logic_df in grouped_by_logic:
        fig = go.Figure()

        for method, method_df in logic_df.groupby("method"):
            # Calculate mean, std, min, and max for all iterations
            summary = method_df.groupby("iteration")["error"].agg(["mean", "std", "min", "max"]).reset_index()

            # Mean curve with markers
            fig.add_trace(go.Scatter(
                x=summary["iteration"],
                y=summary["mean"],
                mode="lines+markers",
                name=f"{method} (mean)",
                line=dict(width=2),
            ))

            # Shaded area for min-max range
            fig.add_trace(go.Scatter(
                x=pd.concat([summary["iteration"], summary["iteration"][::-1]]),
                y=pd.concat([summary["max"], summary["min"][::-1]]),
                fill="toself",
                fillcolor="rgba(0, 100, 80, 0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name=f"{method} (min-max range)"
            ))

            # Error bars for standard deviation
            fig.add_trace(go.Scatter(
                x=summary["iteration"],
                y=summary["mean"],
                mode="markers",
                error_y=dict(
                    type='data',
                    array=summary["std"],
                    visible=True,
                    color='gray'
                ),
                marker=dict(opacity=0),
                showlegend=False
            ))

        # Add convergence threshold line
        fig.add_trace(go.Scatter(
            x=[0, max_iteration],
            y=[convergence_threshold, convergence_threshold],
            mode="lines",
            name=f"Convergence Threshold (error = {convergence_threshold})",
            line=dict(color="black", width=2, dash="dash"),
            hoverinfo="skip"
        ))

        # Update layout
        fig.update_layout(
            title=f"Convergence Plot for {logic_gate} Gate",
            xaxis_title="Iteration",
            yaxis_title="Error",
            template="plotly_white",
            hovermode="x unified"
        )

        # Save plots
        safe_gate_name = logic_gate.replace(" ", "_").lower()
        html_path = os.path.join(output_folder, f"{safe_gate_name}_interactive_plot.html")
        static_path = os.path.join(output_folder, f"{safe_gate_name}_plot.png")

        fig.write_html(html_path)
        pio.write_image(fig, static_path, format="png")

        print(f"Plots saved for {logic_gate}:\n  Interactive: {html_path}\n  Static PNG : {static_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate convergence plots from CSV data.")
    parser.add_argument("root", type=str, help="Root directory containing subdirectories with CSV files")
    parser.add_argument("-s", "--subdirs", nargs="+", required=True, help="List of subdirectories to include")
    return parser.parse_args()

def main():
    args = parse_arguments()
    folder_paths = [os.path.join(args.root, subdir) for subdir in args.subdirs]
    convergence_threshold = 0.25

    full_df, max_iteration = load_and_prepare_data(folder_paths, convergence_threshold)
    generate_and_save_plots(full_df, args.root, convergence_threshold, max_iteration)

if __name__ == "__main__":
    main()