import os
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

def main():
    try:
        # Check if required CSV files exist
        if not os.path.exists("output.csv"):
            print("Error: output.csv not found")
            return
        
        if not os.path.exists("first_half.csv"):
            print("Error: first_half.csv not found")
            return
            
        if not os.path.exists("second_half.csv"):
            print("Error: second_half.csv not found")
            return
        
        # Read date from first_half.csv and create directory
        with open("first_half.csv", 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            if len(rows) < 2:
                print("Error: first_half.csv doesn't have at least 2 rows")
                return
            
            date_str = rows[1][0]  # First column of second row
            
            # Parse and format the date for directory name
            try:
                # Try parsing with the format you provided
                date_obj = datetime.strptime(date_str, '%d/%m/%Y  %H:%M:%S')
                dir_name = date_obj.strftime('%Y-%m-%d_%H-%M-%S')
            except ValueError:
                # If that format fails, try some common alternatives
                try:
                    date_obj = datetime.strptime(date_str, '%d/%m/%Y %H:%M:%S')
                    dir_name = date_obj.strftime('%Y-%m-%d_%H-%M-%S')
                except ValueError:
                    # If parsing fails, use a sanitized version of the original string
                    dir_name = date_str.replace('/', '-').replace(':', '-').replace(' ', '_')
        
        # Get parent directory from command line arguments
        parent_dir = None
        if len(sys.argv) >= 7:
            parent_dir = sys.argv[6]
        
        # Create the full directory path
        if parent_dir:
            full_dir_path = os.path.join(parent_dir, dir_name)
            os.makedirs(parent_dir, exist_ok=True)  # Ensure parent directory exists
        else:
            full_dir_path = dir_name
        
        os.makedirs(full_dir_path, exist_ok=True)
        print(f"Created directory: {full_dir_path}")
        
        # Get values from command line arguments
        if len(sys.argv) >= 5:
            model = sys.argv[1]
            gpus = sys.argv[2]
            cpus = sys.argv[3]
            node = sys.argv[4]
        else:
            # Fallback to environment variables if arguments not provided
            model = os.environ.get('MODEL', '')
            gpus = os.environ.get('GPUS', '')
            cpus = os.environ.get('CPUS', '')
            node = os.environ.get('NODE', '')
        
        # Get stage from command line arguments or environment
        if len(sys.argv) >= 6:
            stage = sys.argv[5]
        else:
            stage = os.environ.get('STAGE', '')
        
        # Get values from .env file
        min_input_tokens = ''
        min_output_tokens = ''
        req_min = ''
        
        if os.path.exists('../.env'):
            with open('../.env', 'r') as env_file:
                for line in env_file:
                    line = line.strip()
                    if line.startswith('MIN_INPUT_TOKENS='):
                        min_input_tokens = line.split('=', 1)[1]
                    elif line.startswith('MIN_OUTPUT_TOKENS='):
                        min_output_tokens = line.split('=', 1)[1]
                    elif line.startswith('REQ_MIN='):
                        req_min = line.split('=', 1)[1]
        
        # Get evaluation from output.csv
        evaluation = ''
        with open("output.csv", 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            if len(rows) > 0:
                # Find the column index for "no_statistical_difference"
                header = rows[0]
                if 'no_statistical_difference' in header:
                    no_statistical_difference_index = header.index('no_statistical_difference')
                    if len(rows) > 1:
                        evaluation = rows[1][no_statistical_difference_index] if len(rows[1]) > no_statistical_difference_index else ''

        # Create new CSV file
        new_csv_path = os.path.join(full_dir_path, "results.csv")
        with open(new_csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write header with new STAGE column
            writer.writerow(["MODEL", "GPUS", "CPUS", "NODE", "STAGE", "INPUT_TOKENS", "OUTPUT_TOKENS", "EVALUATION", "REQ_MIN"])
            # Write data row with stage value
            writer.writerow([model, gpus, cpus, node, stage, min_input_tokens, min_output_tokens, evaluation, req_min])
        
        print(f"Created results.csv in {full_dir_path}")
        
        # Move CSV files to the new directory
        shutil.move("output.csv", os.path.join(full_dir_path, "output.csv"))
        shutil.move("first_half.csv", os.path.join(full_dir_path, "first_half.csv"))
        shutil.move("second_half.csv", os.path.join(full_dir_path, "second_half.csv"))
        shutil.move("results.json", os.path.join(full_dir_path, "results.json"))
        
        print("Moved all CSV files to the directory")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()