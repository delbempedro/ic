#!/usr/bin/env python3
"""
analize_convergence.py

Generate a convergence summary table from CSV files in subfolders.

Usage:
    python analize_convergence.py

The root_folder should contain subfolders named by encoding,
each containing CSV files with the format:
    Logic Gate; Method; Iteration; Error; Run

Output:
    max_iterations_summary.csv
"""
import os
import csv
import argparse

def max_iterations_to_converge(data):
    try:
        max_iter = max(int(row['Iteration']) for row in data)
        return max_iter
    except Exception as e:
        print(f"Erro ao calcular max_iterations: {e}")
        return 0

def parse_gate_from_data(data):
    if len(data) == 0:
        return None
    # Assume que o logic gate é o mesmo para todo arquivo
    return data[0].get('Logic Gate', None)

def parse_method_from_filename(filename):
    # Supondo que o método está no nome do arquivo, entre _ ou -
    # Exemplo: AND_cg-exhaustive.csv -> cg-exhaustive
    parts = filename.lower().replace('.csv', '').split('_')
    if len(parts) >= 2:
        return parts[1]
    # Se não encontrar, tenta outro split por hífen
    parts = filename.lower().replace('.csv', '').split('-')
    if len(parts) >= 2:
        return parts[1]
    return None

def main():
    parser = argparse.ArgumentParser(description='Analisar convergência dos resultados')
    parser.add_argument('folder', type=str, help='Pasta com arquivos CSV de resultados (ex: amplitude-2-inputs)')
    args = parser.parse_args()

    folder = 'results/' + args.folder
    if not os.path.isdir(folder):
        print(f"Pasta {folder} não existe.")
        return

    output_filename = f'convergence_{args.folder}.csv'

    with open(output_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Logic Gate', 'Method', 'Encoding', 'Max Iterations to Converge'])

        for filename in os.listdir(folder):
            if not filename.endswith('.csv'):
                continue
            filepath = os.path.join(folder, filename)

            with open(filepath, newline='') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=';')
                # Remove espaços extras das chaves do dicionário
                data = [{k.strip(): v for k, v in row.items()} for row in reader]

            logic_gate = parse_gate_from_data(data)
            method = parse_method_from_filename(filename)
            encoding = args.folder  # encoding vem do nome da pasta

            if logic_gate is None or method is None:
                print(f"Pulando arquivo {filename}: não conseguiu extrair logic gate ou método.")
                continue

            max_iter = max_iterations_to_converge(data)

            writer.writerow([logic_gate, method, encoding, max_iter])

    print(f"Arquivo gerado: {output_filename}")

if __name__ == '__main__':
    main()


