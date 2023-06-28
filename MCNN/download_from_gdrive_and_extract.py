import gdown, os

url_start = 'https://drive.google.com/uc?id='
# ids = [
#     '1U8N4X-Snkx9EYk_UvFghxmfS-qmdxCjj', 
#     '106YQ0oDcz5UaYM45mDQA3LeBUgQpawOU',
#     '1hz6UNWpQHlh3kMohlKYxzqDt89RB_xkL',
#     '1itoUjeBe9jURvfqeczPxjHVIx2T0G2F3',
#     '1cdyIB9nu57RqDzElz6HUsZ4AR78iTq0V',
#     '1BIzpdUOWE9dFr4cUlEdwO-Rtalf2fZw3',
# ]

ids = [
    '1LZQBbEDMD-gb7aEgCr5DwbqTBEj79W7s', # pouring_parsed_with_embeddings_moco_conv5.pkl
    '1nMsdc6yOMA4vcrnQqfkPZ2JAV2e_HzXq', # scooping_parsed_with_embeddings_moco_conv5.pkl 
]

os.chdir('assets')
for id in ids:
    url = f'{url_start}{id}'
    gdown.download(url, quiet=False)

# # unzip the following files: assets/cloud-data-pouring.zip, assets/cloud-data-scooping.zip, assets/data_samples.zip with os
# for f in ['cloud-data-pouring.zip', 'data_samples.zip']:
#     print(f'unzipping {f}')
#     os.system(f'unzip {f}')
# print(f'unzipping cloud-data-scooping.zip with different command because of some issue with the zip file')
# os.system(f'jar xvf cloud-data-scooping.zip')

