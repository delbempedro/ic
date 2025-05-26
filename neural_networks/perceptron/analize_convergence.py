#!/usr/bin/env python3
"""
analize_convergence.py

Generate a convergence summary table from CSV files in subfolders.

Usage:
    python analize_convergence.py

The 'results/' directory should contain subfolders named by encoding
(e.g., 'amplitude-2-inputs', 'phase-2-inputs', etc.), and each encoding folder
should contain multiple CSV files, one per logic gate and optimization method.

Each CSV file must have the following format (semicolon-separated):
    Logic Gate; Method; Iteration; Error; Run

This script analyzes each CSV file and computes, for each combination of:
(logic gate, optimization method, encoding), the **maximum number of iterations**
needed to converge among all runs. The result is written to:

    convergence_all.csv
"""

import os
import csv

# Dictionary to map short method names (from filenames) to full names
METHOD_NAME_MAP = {
    'cg': 'Exhaustive Grid Search',
    'genetic': 'Genetic Algorithm',
    'gradient': 'Gradient Descent',
    'random': 'Random Search',
    'simulated': 'Simulated Annealing'
}

def max_iterations_to_converge(data):
    """
    Computes the maximum iteration number reached across all runs.
    """
    try:
        return max(int(row['Iteration']) for row in data)
    except Exception as e:
        print(f"Error computing max_iterations: {e}")
        return 0

def parse_gate_from_data(data):
    """
    Extracts the logic gate name from the first row of the CSV data.
    """
    if len(data) == 0:
        return None
    return data[0].get('Logic Gate', None)

def parse_method_from_filename(filename):
    """
    Extracts the optimization method from the filename and returns the full name.
    """
    name = filename.lower().replace('.csv', '')
    for key in METHOD_NAME_MAP:
        if key in name:
            return METHOD_NAME_MAP[key]
    return None

def analyze_all_results(results_dir='results'):
    """
    Walks through the results directory and aggregates convergence information
    from all CSV files into a summary table.
    """
    output_filename = 'convergence_all.csv'

    with open(output_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Logic Gate', 'Method', 'Encoding', 'Max Iterations to Converge'])

        # Iterate through all encoding folders
        for encoding_folder in os.listdir(results_dir):
            encoding_path = os.path.join(results_dir, encoding_folder)
            if not os.path.isdir(encoding_path):
                continue  # Skip non-directory files

            # Iterate through all CSV files in the encoding folder
            for filename in os.listdir(encoding_path):
                if not filename.endswith('.csv'):
                    continue

                filepath = os.path.join(encoding_path, filename)

                try:
                    with open(filepath, newline='') as csvfile:
                        reader = csv.DictReader(csvfile, delimiter=';')
                        data = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]

                    logic_gate = parse_gate_from_data(data)
                    method = parse_method_from_filename(filename)
                    encoding = encoding_folder

                    # Skip if essential information is missing
                    if logic_gate is None or method is None:
                        print(f"Skipping file {filename}: missing logic_gate or method.")
                        continue

                    max_iter = max_iterations_to_converge(data)

                    # Write the result for this file
                    writer.writerow([logic_gate, method, encoding, max_iter])

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    print(f"Output file generated: {output_filename}")

if __name__ == '__main__':
    analyze_all_results()
