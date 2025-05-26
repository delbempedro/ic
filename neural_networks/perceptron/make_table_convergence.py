#!/usr/bin/env python3
"""
make_table_convergence.py

Generate convergence summary heatmaps grouped by input suffix,
showing logic gates once on y-axis and a separate legend for encodings.

Usage:
    python make_table_convergence.py [convergence_all.csv]

Output:
    convergence_X-inputs.png for each group X
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys
import textwrap

def extract_input_suffix(encoding):
    match = re.search(r'(\d+-inputs)$', encoding)
    return match.group(1) if match else None

def clean_encoding(encoding):
    return re.sub(r'-\d+-inputs$', '', encoding)

def wrap_labels(labels, width=14):
    return ['\n'.join(textwrap.wrap(label, width)) for label in labels]

def plot_grouped_heatmap(pivot_table, input_suffix):
    """
    Plot heatmap showing logic gates once on y-axis,
    and separate legend for encodings (rows per logic gate).
    """
    plt.figure(figsize=(12, max(4, 0.5 * len(pivot_table))))

    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        cbar=False,
        linewidths=0.5,
        linecolor='gray',
        yticklabels=False  # We'll add logic gate labels manually
    )

    logic_gates = pivot_table.index.get_level_values(0)
    encodings = pivot_table.index.get_level_values(1)
    methods = pivot_table.columns

    # Group by logic gate to find the line indices
    from itertools import groupby
    groups = []
    for logic_gate, group_iter in groupby(enumerate(logic_gates), key=lambda x: x[1]):
        indices = [i for i, _ in group_iter]
        groups.append((logic_gate, indices))

    # Write logic gate labels once at center of their group, aligned left
    for logic_gate, indices in groups:
        mid_pos = sum(indices) / len(indices)
        ax.text(
            -0.5,
            mid_pos + 0.5,
            logic_gate,
            ha='right',
            va='center',
            fontsize=10,
            fontweight='bold',
            clip_on=False,
            rotation=0
        )

    # Wrap method names horizontally at top
    wrapped_methods = wrap_labels(methods.tolist(), width=16)
    ax.set_xticklabels(wrapped_methods, rotation=0, ha='center')

    # Legend for encodings (e.g., amplitude, phase)
    unique_encodings = sorted(encodings.unique())
    encoding_legend_text = "\n".join([f"{i+1} = {enc}" for i, enc in enumerate(unique_encodings)])

    # Put legend box to the right of the plot
    plt.gcf().text(
        0.95, 0.5, encoding_legend_text,
        fontsize=10,
        va='center',
        ha='left',
        bbox=dict(boxstyle="round,pad=0.5", fc="lightgray", ec="gray", lw=1)
    )

    plt.xlabel("Training Method")
    plt.ylabel("")
    plt.title(f"Convergence Summary - {input_suffix}")
    plt.tight_layout(rect=[0, 0, 0.92, 1])  # Leave space for legend on right

    # Adjust x and y limits so labels show correctly
    ax.set_xlim(-0.7, len(methods))
    ax.set_ylim(len(pivot_table), 0)

    return plt, ax

def main(csv_file='convergence_all.csv'):
    try:
        df = pd.read_csv(csv_file)
        df.columns = [col.strip() for col in df.columns]

        df['Input Suffix'] = df['Encoding'].apply(extract_input_suffix)
        df = df[df['Input Suffix'].notna()]
        df['Encoding Clean'] = df['Encoding'].apply(clean_encoding)

        df['Method'] = df['Method'].str.strip().str.lower().str.title()

        for input_suffix in sorted(df['Input Suffix'].unique()):
            df_group = df[df['Input Suffix'] == input_suffix].copy()

            pivot_table = df_group.pivot_table(
                index=['Logic Gate', 'Encoding Clean'],
                columns='Method',
                values='Max Iterations to Converge',
                aggfunc='max'
            )

            pivot_table = pivot_table.sort_index()

            plt_obj, ax = plot_grouped_heatmap(pivot_table, input_suffix)

            filename = f"convergence_{input_suffix}.png"
            plt_obj.savefig(filename, bbox_inches='tight')
            plt_obj.close()
            print(f"Saved image: {filename}")

    except FileNotFoundError:
        print(f"File '{csv_file}' not found.")
    except Exception as e:
        print(f"Error while processing: {e}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
