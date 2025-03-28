import os
import requests
import tarfile
import gzip
import shutil
from tqdm import tqdm


def download_tar_extract(url, download_path, extract=True):
    tar_file_name = url.split('/')[-1]
    extract_folder_name = tar_file_name.split('.')[0]
    
    if os.path.exists(os.path.join(download_path, extract_folder_name)):
        print(f'Found {extract_folder_name} already installed')
        return
    
    if not os.path.exists(os.path.join(download_path, tar_file_name)):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
    
        with open(os.path.join(download_path, tar_file_name), 'wb') as f, tqdm(
            desc=f'Downloading {tar_file_name}', total=total_size, unit='B', unit_scale=True
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                pbar.update(len(data))
    
    if extract:
        if '.'.join(tar_file_name.split('.')[-2:]) == 'tar.gz' or \
            tar_file_name.split('.')[-1] in ['tar', 'tgz']:
            with tarfile.open(os.path.join(download_path, tar_file_name), 'r') as tar:
                total_files = len(tar.getnames())
                with tqdm(total=total_files, desc=f'Extracting {tar_file_name}', unit='file') as pbar:
                    for member in tar.getmembers():
                        tar.extract(member, path=download_path)
                        pbar.update(1)
        else:
            raise NotImplementedError
    
        os.remove(os.path.join(download_path, tar_file_name))