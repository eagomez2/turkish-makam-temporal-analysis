from tqdm import tqdm
import os
import requests
import zipfile

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_FILE = 'SymbTr-2.4.3.zip'
DATASET_URL = 'https://github.com/MTG/SymbTr/archive/v2.4.3.zip'
DATASET_DIR = os.path.join(BASE_DIR, 'code', 'dataset')

print(f'Downloading dataset from {DATASET_URL}...')

response = requests.get(DATASET_URL, stream=True)
total_size_in_bytes = int(response.headers.get('content-length', 0))
block_size = 1024
progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

with open(os.path.join(DATASET_DIR, DATASET_FILE), 'wb') as file:
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        file.write(data)

progress_bar.close()

if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
    print('An error ocurred while download the files. Please run this script again.')
else:
    print('Dataset successfully downloaded')
    print(f'Uncompressing {DATASET_FILE}...')
    zip_extractor = zipfile.ZipFile(os.path.join(DATASET_DIR, DATASET_FILE), 'r')
    zip_extractor.extractall(DATASET_DIR)
    zip_extractor.close()

    print(f'File successfully uncompressed')
    print(f'Deleting original file {DATASET_FILE}...')
    os.remove(os.path.join(DATASET_DIR, DATASET_FILE))

    print(f'Dataset successfully saved to {os.path.join(DATASET_DIR, os.path.splitext(DATASET_FILE)[0])}')
