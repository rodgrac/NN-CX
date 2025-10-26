import os
import requests
import tarfile
import zipfile
from tqdm import tqdm


def download_extract_file(url, download_path, file_name=None, extract=True):
    if file_name is None:
        file_name = url.split('/')[-1]
    
    folder_name = file_name.split('.')[0]
    
    if os.path.exists(os.path.join(download_path, folder_name)):
        print(f'Found {folder_name} already installed')
        return
    
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
        elif file_name.split('.')[-1] == 'zip':
            with zipfile.ZipFile(os.path.join(download_path, file_name), 'r') as zip:
                total_files = len(zip.namelist())
                with tqdm(total=total_files, desc=f'Extracting {file_name}', unit='file') as pbar:
                    for member in zip.namelist():
                        zip.extract(member, path=download_path)
                        pbar.update(1)
        else:
            raise NotImplementedError
    
        os.remove(os.path.join(download_path, file_name))
