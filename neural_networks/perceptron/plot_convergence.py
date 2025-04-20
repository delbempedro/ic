"""
plot_convergence.py

Module to plot the convergence of qNN trainin methods.

Dependencies:
- Use matplotlib to plot the convergence graphs
- Use re to parse the results file
- Use sys to handle command line argumets
- Use os to handle file paths

Since:
- 04/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""
import matplotlib.pyplot as plt # type: ignore
import re
import os

# Function to parse error histories from the results file
def parse_error_history(filepath):
    """
    Parses the error history from a results file.

    This function reads a given file line by line, extracting the error history for each training method. 
    It identifies lines that indicate the start of a new training method and records subsequent errors 
    associated with that method. The result is a dictionary where keys are training method names and values 
    are lists of error values for each iteration.

    Parameters:
    filepath (str): The path to the results file containing error histories.

    Returns:
    dict: A dictionary mapping each training method to its list of error values.
    """

    error_history = {}
    current_method = None

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()

            # Match training method line (handles underscores)
            method_match = re.search(r"Training with (.+?) method", line)
            if method_match:
                current_method = method_match.group(1).strip()
                error_history[current_method] = []
                continue

            # Match iteration error lines
            error_match = re.search(r"Iteration\s+\d+:\s+Error:\s+([0-9.]+)", line)
            if error_match and current_method:
                error = float(error_match.group(1))
                error_history[current_method].append(error)

    return error_history

# Function to plot the convergence for each file
def plot_convergence_for_directory(input_dir, output_dir):
    """
    Plots the convergence of qNN training methods for each file in the given input directory.

    Parameters:
    input_dir (str): The directory containing the .txt files with the training results.
    output_dir (str): The directory where the plots should be saved.

    Returns:
    None
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            error_history = parse_error_history(input_path)

            # Create the plot
            plt.figure(figsize=(10, 6))
            for method, errors in error_history.items():
                plt.plot(range(1, len(errors) + 1), errors, label=method)

            title = make_title(filename)

            plt.xlabel('Iteration')
            plt.ylabel('Error')
            plt.title(title)
            plt.legend()
            plt.grid(True)

            # Save the plot to the output directory
            output_filename = f"{os.path.splitext(filename)[0]}.png"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path)
            plt.close()

            print(f"Plot saved to {output_path}")

def make_title(filename):

    parts_of_filename = filename.split("_")
    # Remove the file extension
    parts_of_filename[-1] = parts_of_filename[-1].split(".")[0]
    # Remove the first part of the filename
    parts_of_filename.pop(0)
    # Join the remaining parts with spaces
    final_title = "Enconding: "+parts_of_filename[0]+" | "+"Logic Gate: "+parts_of_filename[1]+" | "+ "Tolerance: "+parts_of_filename[2]+" | "+"Grid Grain: "+parts_of_filename[3]+" | "

    return f"Convergence Plot for {final_title}"

def main():
    """
    Asks the user for input and output directories and plots the convergence of qNN training methods for each file in the given input directory.

    Parameters:
    None

    Returns:
    None
    """
    # Ask the user for input and output directories
    input_dir = input("Enter the directory containing the .txt files: ")
    output_dir = input("Enter the directory where the plots should be saved: ")
    if input_dir == "":
        input_dir = "results"
    if output_dir == "":
        output_dir = "plots"
    plot_convergence_for_directory(input_dir, output_dir)

# Main function to execute the script
if __name__ == "__main__":
    main()