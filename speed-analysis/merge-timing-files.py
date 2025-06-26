import json
import csv
import os
import glob

def create_benchmark_csv(json_directory='./timing_files', output_csv_file='benchmark_summary.csv'):
    """
    Reads all JSON benchmark files from a directory, flattens the data,
    and writes it to a single CSV file.

    Args:
        json_directory (str): Directory containing the JSON files. Defaults to the
                              current directory.
        output_csv_file (str): Name for the output CSV file. Defaults to
                               'benchmark_summary.csv'.
    """
    # Use glob to find all files ending with .json in the specified directory
    json_files = glob.glob(os.path.join(json_directory, '*.json'))

    if not json_files:
        print(f"No JSON files found in '{os.path.abspath(json_directory)}'.")
        return

    print(f"Found {len(json_files)} JSON files. Processing...")
    all_rows = []

    # Loop through each JSON file found
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # The system_info is common for all benchmarks in this file
            system_info = data.get('system_info', {})
            # Add the source filename for traceability
            system_info['source_file'] = os.path.basename(file_path)

            # Iterate through each benchmark type (e.g., 'float64_inverse')
            benchmark_results = data.get('benchmark_results', {})
            for benchmark_name, results_data in benchmark_results.items():
                
                # Iterate through each measurement [size, time]
                for result_pair in results_data.get('results', []):
                    if len(result_pair) == 2:
                        matrix_size, time = result_pair
                        
                        # Create a dictionary for the CSV row
                        row_data = {
                            'benchmark_name': benchmark_name,
                            'matrix_size': str(matrix_size), # Convert matrix size to string to handle nested lists
                            'time_seconds': time,
                        }
                        # Add all the system info to this row
                        row_data.update(system_info)
                        all_rows.append(row_data)

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{file_path}'. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred with file '{file_path}': {e}")

    if not all_rows:
        print("No valid benchmark data could be extracted.")
        return

    # Write the collected data to a CSV file
    try:
        # Collect all unique headers from all rows to prevent errors
        headers = set()
        for row in all_rows:
            headers.update(row.keys())
        
        # For consistency, sort the headers alphabetically
        ordered_headers = sorted(list(headers))

        if all_rows:
            with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=ordered_headers, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(all_rows)
            
            print(f"\nSuccess! CSV file created: '{output_csv_file}'")
            print(f"Total rows written: {len(all_rows)}")
        else:
            print("No data to write to CSV.")

    except Exception as e:
        print(f"An error occurred while writing the CSV file: {e}")


if __name__ == '__main__':
    # --- How to use ---
    # 1. Save this code as a Python file (e.g., process_benchmarks.py).
    # 2. Place this script in the same folder as your JSON files.
    # 3. Run the script from your terminal: python process_benchmarks.py
    #
    # The script will automatically find all .json files in its directory
    # and create a 'benchmark_summary.csv' file.
    create_benchmark_csv()