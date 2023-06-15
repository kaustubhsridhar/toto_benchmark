import gdown, os

url_start = 'https://drive.google.com/uc?id='
ids = [
    '1U8N4X-Snkx9EYk_UvFghxmfS-qmdxCjj',
    '106YQ0oDcz5UaYM45mDQA3LeBUgQpawOU',
    '1hz6UNWpQHlh3kMohlKYxzqDt89RB_xkL',
    '1itoUjeBe9jURvfqeczPxjHVIx2T0G2F3',
    '1cdyIB9nu57RqDzElz6HUsZ4AR78iTq0V',
    '1BIzpdUOWE9dFr4cUlEdwO-Rtalf2fZw3',
]

os.chdir('assets')
for id in ids:
    url = f'{url_start}{id}'
    gdown.download(url, quiet=False)

# unzip the following files: assets/cloud-data-pouring.zip, assets/cloud-data-scooping.zip, assets/data_samples.zip with os
for f in ['cloud-data-pouring.zip', 'cloud-data-scooping.zip', 'data_samples.zip']:
    print(f'unzipping {f}')
    os.system(f'unzip {f}')

