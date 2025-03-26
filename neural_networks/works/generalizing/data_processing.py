"""
  data_processing.py

Module that converts np.float64(x) to x/pi.

Dependencies:
- Uses regular expressions to find and replace np.float64(x)
- Uses numpy to convert x to x/pi

Since:
- 03/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""
import re
import numpy as np  # type: ignore

def convert_to_pi(value):
    """
    Convert a value to a string in the form x/pi.

    Parameters
    ----------
    value : str or float
        The value to be converted.

    Returns
    -------
    str
        The value as a string in the form x/pi.
    """
    try:
        result = float(value) / np.pi
        if result == 1:
            return "π"
        elif result == -1:
            return "-π"
        elif result == 0:
            return "0"
        else:
            # Remove trailing zeros and ensure that the result is not something like 0.500000
            return f"{result:.6f}".rstrip('0').rstrip('.') + "π"
    except ValueError:
        return value

def process_file(input_file, output_file):
    """
    Process the input file and replace np.float64(x) with x/pi.

    Parameters
    ----------
    input_file : str
        The name of the input file.
    output_file : str
        The name of the output file.

    Returns
    -------
    None
    """
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Regex to find np.float64(x)
    pattern = re.compile(r'np\.float64\((-?[0-9\.]+)\)')
    
    # Function to replace np.float64(x) with x/pi
    def replace_float(match):
        value = match.group(1)
        return convert_to_pi(value)
    
    # Replace np.float64(x) with x/pi
    updated_content = pattern.sub(replace_float, content)
    
    # Write the updated content to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(updated_content)


def main():
    """
    Reads an input file name and an output file name from the user,
    and processes the input file, replacing np.float64(x) with x/pi,
    and writes the result to the output file.
    """
    # Ask user for the input file name
    print("Enter the name of the input file:")

    # Get the input file name
    input_file = input()

    # Define the output file name
    output_file = "treated_" + input_file

    # Process the file
    process_file(input_file, output_file)

if __name__ == "__main__":
    main()
