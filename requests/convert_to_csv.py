import json
import csv
import argparse
from collections import defaultdict
from datetime import datetime

def calculate_completion_time_and_success(json_file_path="results.json", output_csv_path="output.csv"):
    """
    Calculate completion time and success rate for each request group, then export to CSV.

    Args:
        json_file_path: Path to the JSON file containing the response data
        output_csv_path: Path where the CSV file will be saved

    Returns:
        Dictionary with request_idx as keys and completion times as values
    """
    # Load data from JSON file
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)

    # Extract the results array from the JSON data
    # Check if the JSON has a "results" key, otherwise use the data directly
    if "results" in json_data:
        results_data = json_data["results"]
    else:
        results_data = json_data

    # Group by request_idx
    request_groups = defaultdict(list)

    for item in results_data:
        request_idx = item["request_idx"]
        request_groups[request_idx].append(item)

    # Calculate completion time and success for each request group
    completion_data = {}
    total_requests = len(request_groups)
    successful_requests = 0

    for request_idx, group in request_groups.items():
        # Sort the group by timestamp to ensure chronological order
        group.sort(key=lambda x: x["timestamp"])

        # Calculate start and end times
        start = group[0]["timestamp"] - (group[0]["duration_ms"] * 1_000_000)  # Convert ms to nanoseconds
        end = group[-1]["timestamp"]

        # Calculate completion time in milliseconds
        completion_time_ms = (end - start) / 1_000_000  # Convert nanoseconds to milliseconds
        
        # Check if request is successful (all tokens have ok=True and error="None")
        is_successful = all(item["ok"] and item["error"] == "None" for item in group)
        if is_successful:
            successful_requests += 1

        # Convert end timestamp (assumed to be in nanoseconds since epoch) to formatted string
        end_dt = datetime.utcfromtimestamp(end / 1_000_000_000)
        received_timestamp_str = end_dt.strftime('%Y%m%dT%H%M%S')

        completion_data[request_idx] = {
            "complete_response_time": completion_time_ms,
            "received_timestamp": received_timestamp_str,
            "success": is_successful
        }

    # Calculate overall success rate (as percentage)
    success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0

    # Sort the data by received_timestamp
    sorted_data = sorted(completion_data.values(), key=lambda x: x["received_timestamp"])

    # Generate CSV file
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['received_timestamp', 'complete_response_time', 'success_rate', 'success']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for data in sorted_data:
            writer.writerow({
                'received_timestamp': data['received_timestamp'],
                'complete_response_time': data['complete_response_time'],
                'success_rate': success_rate,  # Same for all rows
                'success': data['success']  # Individual request success status
            })

    print(f"CSV file generated successfully at: {output_csv_path}")
    print(f"Total requests: {total_requests}")
    print(f"Successful requests: {successful_requests}")
    print(f"Success rate: {success_rate:.2f}%")

    return completion_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON requests data to CSV with completion times and success rate.")
    parser.add_argument("json_file_path", nargs="?", default="results.json", help="Path to the input JSON file (default: result.json)")
    parser.add_argument("output_csv_path", nargs="?", default="output.csv", help="Path to the output CSV file (default: output.csv)")
    args = parser.parse_args()

    calculate_completion_time_and_success(args.json_file_path, args.output_csv_path)