import pandas as pd
import os
import sys

def get_top_parameters(csv_path, output_folder):
    # Read the CSV file
    df = pd.read_csv(csv_path, delimiter=';')

    # Sort by last_task_accuracy in descending order
    df = df.sort_values(by='last_task_accuracy', ascending=False)

    # Drop duplicates based on last_task_accuracy, keeping the first occurrence
    df = df.drop_duplicates(subset='last_task_accuracy')

    # Get the original file name
    original_name = os.path.basename(csv_path)

    # Create the output file name
    output_file = os.path.join(output_folder, f'BEST_{original_name}')

    # Save the filtered DataFrame to the output file
    df.to_csv(output_file, index=False, sep=';')

if __name__ == "__main__":
    # Example usage:
    # python get_top_parameters.py path/to/input.csv path/to/output/folder

    # Get the command line arguments
    csv_path = sys.argv[1]
    output_folder = sys.argv[2]

    # Call the function to get top parameters and save the result
    get_top_parameters(csv_path, output_folder)
