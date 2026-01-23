import os
import pandas as pd
import json

def get_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

# Create ingress_bytes folder if it doesn't exist
output_folder = 'ingress_bytes'
os.makedirs(output_folder, exist_ok=True)

folder = '../../Millisampler-data-main/day1-h1-zip'
files = [f for f in os.listdir(folder) if f.endswith('.txt')]

num_good_files = 0
num_bad_files = 0

for file in files:
    file_path = os.path.join(folder, file)
    print(f'Processing file: {file_path}')
    
    try:
        content = json.loads(get_file(file_path))
        
        # Check if ingressBytes exists
        if 'ingressBytes' not in content:
            print(f'Skipping file (no ingressBytes): {file_path}')
            num_bad_files += 1
            continue
        
        rack_id = content.get('rack_id', 'unknown')
        host_id = content.get('server_hostname', 'unknown')
        ingress_bytes = content['ingressBytes']
        
        # Get start timestamp from burst_result if available
        start_timestamp = None
        if 'burst_result' in content and 'ingress' in content['burst_result']:
            timestamps = list(content['burst_result']['ingress'].keys())
            if timestamps:
                start_timestamp = int(timestamps[0]) / 1000  # Convert to milliseconds
        
        # Get sampling frequency
        sampling_freq = content.get('sampling_freq', 1000)
        
        # Create DataFrame with ingress bytes and sample index
        df = pd.DataFrame({
            'sample_index': range(len(ingress_bytes)),
            'ingressBytes': ingress_bytes
        })
        
        # Add metadata columns
        df['rack_id'] = rack_id
        df['host_id'] = host_id
        df['sampling_freq'] = sampling_freq
        if start_timestamp:
            df['start_timestamp_ms'] = start_timestamp
        
        # Save the DataFrame to a CSV file in the ingress_bytes folder
        output_csv_path = os.path.join(output_folder, f'ingress_bytes_rackId_{rack_id}_hostId_{host_id}.csv')
        df.to_csv(output_csv_path, index=False)
        print(f'Data saved to {output_csv_path}')
        num_good_files += 1
        
    except Exception as e:
        print(f'Error processing file {file_path}: {e}')
        num_bad_files += 1

print(f'\nTotal good files processed: {num_good_files}')
print(f'Total bad files skipped: {num_bad_files}')