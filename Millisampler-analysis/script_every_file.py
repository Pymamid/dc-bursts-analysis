# treat files separately and make a dataframe for each file

import os
import pandas as pd
import json
def get_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

folder = '../../Millisampler-data-main/day1-h1-zip'
files = [f for f in os.listdir(folder) if f.endswith('.txt')]

num_good_files = 0
num_bad_files = 0
for file in files:
    file_path = os.path.join(folder, file)
    print(f'Processing file: {file_path}')
    content = json.loads(get_file(file_path))

    burst_records = list(content["burst_result"]["ingress"].values())[0]

    start_timestamp = list(content["burst_result"]["ingress"].keys())[0]

    # Create DataFrame with Position and Length
    if "Position" not in burst_records[0]:
        print(f'Skipping file (no Position): {file_path}')
        num_bad_files += 1
        continue

    df = pd.DataFrame(burst_records)[["Position", "Length"]]
    df.dropna(inplace=True)

    # add a new column with the start timestamp in all rows. convert microseconds to milliseconds
    df['StartTimestamp'] = int(start_timestamp)/1000

    # Save the DataFrame to a CSV file in the individual_csvs folder
    output_csv_path = os.path.join('individual_csvs', f'burst_data_{os.path.splitext(file)[0]}.csv')

    df.to_csv(output_csv_path, index=True)
    print(f'Data saved to {output_csv_path}')
    num_good_files += 1

print('Total good files processed:', num_good_files)
print('Total bad files skipped:', num_bad_files)
