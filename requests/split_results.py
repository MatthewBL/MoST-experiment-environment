import pandas as pd
from datetime import datetime, timedelta

def process_experiment_data(input_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Convert timestamp to datetime
    df['received_timestamp'] = pd.to_datetime(df['received_timestamp'], format='%Y%m%dT%H%M%S')
    
    # Sort by timestamp to ensure chronological order
    df = df.sort_values('received_timestamp').reset_index(drop=True)
    
    # Calculate total experiment duration
    start_time = df['received_timestamp'].min()
    end_time = df['received_timestamp'].max()
    total_duration = end_time - start_time
    
    print(f"Experiment started at: {start_time}")
    print(f"Experiment ended at: {end_time}")
    print(f"Total duration: {total_duration}")
    
    # Remove first and last 5 minutes
    filtered_start = start_time + timedelta(minutes=5)
    filtered_end = end_time - timedelta(minutes=5)
    
    filtered_df = df[
        (df['received_timestamp'] >= filtered_start) & 
        (df['received_timestamp'] <= filtered_end)
    ].copy().reset_index(drop=True)
    
    print(f"\nAfter removing first and last 5 minutes:")
    print(f"Filtered start: {filtered_start}")
    print(f"Filtered end: {filtered_end}")
    print(f"Filtered duration: {filtered_end - filtered_start}")
    print(f"Records in filtered data: {len(filtered_df)}")
    
    # Split into two 10-minute periods
    split_time = filtered_start + timedelta(minutes=10)
    
    first_half = filtered_df[
        filtered_df['received_timestamp'] < split_time
    ].copy().reset_index(drop=True)
    
    second_half = filtered_df[
        filtered_df['received_timestamp'] >= split_time
    ].copy().reset_index(drop=True)
    
    print(f"\nSplit time: {split_time}")
    print(f"First 10-minute period: {len(first_half)} records")
    print(f"Second 10-minute period: {len(second_half)} records")
    
    # Save to new CSV files
    first_half.to_csv('first_half.csv', index=False)
    second_half.to_csv('second_half.csv', index=False)
    
    print(f"\nFiles created:")
    print(f"- first_half.csv ({len(first_half)} records)")
    print(f"- second_half.csv ({len(second_half)} records)")
    
    return first_half, second_half

# Process the data
first_period, second_period = process_experiment_data('output.csv')

# Display some statistics
print(f"\nFirst period statistics:")
print(f"Response time - Min: {first_period['complete_response_time'].min():.2f}, "
      f"Max: {first_period['complete_response_time'].max():.2f}, "
      f"Mean: {first_period['complete_response_time'].mean():.2f}")

print(f"Second period statistics:")
print(f"Response time - Min: {second_period['complete_response_time'].min():.2f}, "
      f"Max: {second_period['complete_response_time'].max():.2f}, "
      f"Mean: {second_period['complete_response_time'].mean():.2f}")