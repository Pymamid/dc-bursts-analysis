import json
def get_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content
file_path_good = '../../Millisampler-data-main/day1-h1-zip/rackId_1156_hostId_106088.txt'
file_path_bad = '../../Millisampler-data-main/day1-h1-zip/rackId_509_hostId_46635.txt'
content_good = json.loads(get_file(file_path_good))
content_bad = json.loads(get_file(file_path_bad))
with open('good.csv', 'w') as good_file:
    good_file.write(json.dumps(content_good, indent=4))
with open('bad.csv', 'w') as bad_file:
    bad_file.write(json.dumps(content_bad, indent=4))

print('Data saved to good.csv and bad.csv')