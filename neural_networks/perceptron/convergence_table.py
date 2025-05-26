#!/usr/bin/env python3
"""
convergence_table.py

Generate a convergence summary table from CSV files in subfolders.

Usage:
    python convergence_table.py <root_folder>

The root_folder should contain subfolders named by encoding,
each containing CSV files named like:
    XOR_simulated_annealing_tolerance_0.25_max_of_iterations_100.csv

Output:
    convergence_summary.csv
"""
import sys
import pandas as pd
from pathlib import Path
import numpy as np
import re

def parse_tolerance(filename):
    # Extrai o valor da tolerância do nome do arquivo, ex: tolerance_0.25
    m = re.search(r'tolerance_([0-9.]+)', filename)
    if m:
        return float(m.group(1))
    return None

def parse_max_iterations(filename):
    # Extrai max_of_iterations do nome do arquivo, ex: max_of_iterations_100
    m = re.search(r'max_of_iterations_([0-9]+)', filename)
    if m:
        return int(m.group(1))
    return None

def main():
    if len(sys.argv) != 2:
        print("Uso: python convergence_table.py <root_folder>")
        sys.exit(1)

    root_folder = Path(sys.argv[1])
    if not root_folder.is_dir():
        print(f"Pasta {root_folder} não encontrada.")
        sys.exit(1)

    all_data = []

    for encoding_folder in root_folder.iterdir():
        if not encoding_folder.is_dir():
            continue
        encoding = encoding_folder.name

        for csv_file in encoding_folder.glob("*.csv"):
            filename = csv_file.stem
            parts = filename.split("_")
            if len(parts) < 2:
                print(f"Arquivo com nome inesperado: {filename}")
                continue

            logic_gate = parts[0]
            method = parts[1]

            try:
                df = pd.read_csv(csv_file, sep=";")
                df.columns = df.columns.str.strip().str.lower()

                required_cols = {"iteration", "error", "run"}
                if not required_cols.issubset(df.columns):
                    print(f"Arquivo {csv_file.name} não tem colunas necessárias.")
                    continue

                df["iteration"] = df["iteration"].astype(int)
                df["error"] = df["error"].astype(float)
                df["run"] = df["run"].astype(int)

                # Adiciona colunas para logic_gate, method, encoding
                df["logic_gate"] = logic_gate
                df["method"] = method
                df["encoding"] = encoding

                # Extrai tolerância e max iterations para usar depois
                tolerance = parse_tolerance(filename)
                max_iter = parse_max_iterations(filename)

                # Armazena dados para análise posterior
                all_data.append({
                    "df": df,
                    "logic_gate": logic_gate,
                    "method": method,
                    "encoding": encoding,
                    "tolerance": tolerance,
                    "max_iter": max_iter
                })
            except Exception as e:
                print(f"Erro lendo {csv_file.name}: {e}")

    # Agora calcula convergência média e taxa por logic_gate, method, encoding

    results = []
    # Organiza os dados agrupando pelas 3 chaves
    from collections import defaultdict
    grouped = defaultdict(list)
    for item in all_data:
        key = (item["logic_gate"], item["method"], item["encoding"], item["tolerance"], item["max_iter"])
        grouped[key].append(item["df"])

    for (logic_gate, method, encoding, tolerance, max_iter), dfs in grouped.items():
        df_concat = pd.concat(dfs, ignore_index=True)

        total_runs = df_concat["run"].nunique()

        # Para cada run, calcula o menor iteration com error < tolerance
        convergences = []
        for run in df_concat["run"].unique():
            df_run = df_concat[df_concat["run"] == run]
            converged_iters = df_run[df_run["error"] <= tolerance]["iteration"]
            if not converged_iters.empty:
                convergences.append(converged_iters.min())
            else:
                # Se não convergiu, considera infinito como max_iter + 1
                if max_iter is not None:
                    convergences.append(max_iter + 1)
                else:
                    convergences.append(np.inf)

        num_converged = sum(1 for c in convergences if c <= (max_iter if max_iter else np.inf))
        convergence_rate = 100 * num_converged / total_runs if total_runs > 0 else 0

        avg_convergence_iter = np.mean(convergences) if convergences else np.nan

        results.append({
            "logic_gate": logic_gate,
            "method": method,
            "encoding": encoding,
            "tolerance": tolerance,
            "max_iterations": max_iter,
            "total_runs": total_runs,
            "num_converged": num_converged,
            "convergence_rate": convergence_rate,
            "avg_convergence_iteration": avg_convergence_iter
        })

    df_result = pd.DataFrame(results)

    # Ordena e salva CSV
    df_result = df_result.sort_values(["logic_gate", "method", "encoding", "tolerance"])
    output_csv = root_folder / "convergence_summary.csv"
    df_result.to_csv(output_csv, index=False)

    print(f"Convergence summary saved to {output_csv}")

if __name__ == "__main__":
    main()
