import os
import pandas as pd
import json

def get_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

folder = '../../Millisampler-data-main/day1-h1-zip'
files = [f for f in os.listdir(folder) if f.endswith('.txt')]

burst_df = pd.DataFrame(columns=["Position", "Length", "StartTimestamp"])

file_num = 0
skip_file_num = 0

for file_path in files:
    file_path = os.path.join(folder, file_path)
    file_num += 1
    print(f'Processing file: {file_path}')
    content = json.loads(get_file(file_path))

    burst_records = list(content["burst_result"]["ingress"].values())[0]

    start_timestamp = list(content["burst_result"]["ingress"].keys())[0]

    # Create DataFrame with Position and Length
    if "Position" not in burst_records[0]:
        skip_file_num += 1
        continue

    df = pd.DataFrame(burst_records)[["Position", "Length"]]
    df.dropna(inplace=True)

    # add a new column with the start timestamp in all rows. convert microseconds to milliseconds
    df['StartTimestamp'] = int(start_timestamp)/1000

    # append df to burst_df
    burst_df = pd.concat([burst_df, df], ignore_index=True)

# Save the DataFrame to a CSV file
burst_df.to_csv('burst_data.csv', index=True)

print('Skipped files:', skip_file_num)
print('Total files processed:', file_num)
print('Data saved to burst_data.csv')