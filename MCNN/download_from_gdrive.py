import gdown, os

url_start = 'https://drive.google.com/uc?id='

ids = [
    '1LZQBbEDMD-gb7aEgCr5DwbqTBEj79W7s', # pouring_parsed_with_embeddings_moco_conv5.pkl
    '1nMsdc6yOMA4vcrnQqfkPZ2JAV2e_HzXq', # scooping_parsed_with_embeddings_moco_conv5.pkl 
    '1BIzpdUOWE9dFr4cUlEdwO-Rtalf2fZw3', # moco_conv5_robocloud.pth
    '1U8N4X-Snkx9EYk_UvFghxmfS-qmdxCjj', # cloud_data_pouring.zip
    '106YQ0oDcz5UaYM45mDQA3LeBUgQpawOU', # cloud_data_scooping.zip
]

os.chdir('assets')
for id in ids:
    url = f'{url_start}{id}'
    gdown.download(url, quiet=False)
