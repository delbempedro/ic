"""
plot_convergence2.py

Module to plot the convergence of qNN trainin methods.

Dependencies:
- Use pandas to organize the data
- Use plotly to plot the convergence graphs
- Use pathlib to handle file paths

Since:
- 04/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""
import pandas as pd # type: ignore
import plotly.graph_objects as go # type: ignore
from pathlib import Path

def main():
    # Ask the user to input the folder path
    folder_path = input("Enter the path to the folder containing the CSV files: ")
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

if __name__ == "__main__":
    main()