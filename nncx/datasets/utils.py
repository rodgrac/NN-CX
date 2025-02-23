import os
import requests
import tarfile
import gzip
import shutil
from tqdm import tqdm


def download_tar_extract(url, download_path, extract=True):
    file_name = url.split('/')[-1]
    
    if not os.path.exists(os.path.join(download_path, file_name)):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
    
        with open(os.path.join(download_path, file_name), 'wb') as f, tqdm(
            desc=f'Downloading {file_name}', total=total_size, unit='B', unit_scale=True
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                pbar.update(len(data))
    
    if extract:
        if '.'.join(file_name.split('.')[-2:]) == 'tar.gz' or \
            file_name.split('.')[-1] in ['tar', 'tgz']:
            with tarfile.open(os.path.join(download_path, file_name), 'r') as tar:
                total_files = len(tar.getnames())
                with tqdm(total=total_files, desc=f'Extracting {file_name}', unit='file') as pbar:
                    for member in tar.getmembers():
                        tar.extract(member, path=download_path)
                        pbar.update(1)
        else:
            raise NotImplementedError
    
        os.remove(os.path.join(download_path, file_name))