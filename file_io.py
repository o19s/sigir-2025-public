import glob
import os
import re
from abc import ABC, abstractmethod
from typing import Iterator, Generator

import boto3


class Fs(ABC):

    """Abstraction of read/write access to local file or S3"""

    @abstractmethod
    def glob(self, pattern: str) -> Generator[str, None, None]:
        pass

    @abstractmethod
    def read(self, path: str) -> bytes:
        pass

    @abstractmethod
    def write(self, path: str, data: bytes) -> None:
        pass
    
    def make_sure_parent_directory_exists(self, file_path: str):
        pass


class LocalFs(Fs):

    def glob(self, pattern: str) -> Generator[str, None, None]:
        for file_path in glob.glob(pattern):
            yield file_path

    def read(self, path: str) -> bytes:
        with open(path, 'rb') as f:
            return f.read()

    def write(self, path: str, data: bytes) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def make_sure_parent_directory_exists(self, file_path: str):
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)


class S3Fs(Fs):

    def __init__(self, bucket_name: str):
        self.s3 = boto3.client('s3')
        self.bucket = bucket_name

    def _parse_s3_path(self, path: str) -> str:
        prefix = f"s3://{self.bucket}/"
        if path.startswith(prefix):
            path = path[len(prefix):]
        return path.lstrip('/')

    def glob(self, pattern: str) -> Iterator[str]:
        s3_pattern = self._parse_s3_path(pattern)

        regex_pattern = re.compile(s3_pattern.replace(".", "\\.").replace("*", ".*").replace("?", "."))

        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket):
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']
                if regex_pattern.fullmatch(key):
                    yield f"s3://{self.bucket}/{key}"

    def read(self, path: str) -> bytes:
        key = self._parse_s3_path(path)
        response = self.s3.get_object(Bucket=self.bucket, Key=key)
        return response['Body'].read()

    def write(self, path: str, data: bytes) -> None:
        key = self._parse_s3_path(path)
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)


def mirror_s3_to_local(s3_url, local_dir, force_resync=False):
    if not s3_url.startswith("s3://"):
        raise ValueError("The S3 URL must start with 's3://'.")
    s3_url = s3_url[5:]
    bucket_name, path = s3_url.split('/', 1)
    last_path_element = path.strip('/').split('/')[-1]
    local_base_dir = os.path.join(local_dir, last_path_element)

    if os.path.isdir(local_base_dir) and not force_resync:
        # assume dir exists = S3 path is mirrored
        print(f"Using existing {local_base_dir}")
        return local_base_dir

    print(f"Downloading {s3_url} to {local_base_dir}")
    if not os.path.exists(local_base_dir):
        os.makedirs(local_base_dir)

    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=path)
    if 'Contents' in response:
        for obj in response['Contents']:
            s3_key = obj['Key']
            local_file_path = os.path.join(local_base_dir, s3_key[len(path):])
            if os.path.exists(local_file_path):
                print(f"Skipping {s3_key}, already exists locally.")
                continue
            local_file_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_file_dir):
                os.makedirs(local_file_dir)
            print(f'Downloading {s3_key} to {local_file_path}')
            s3_client.download_file(bucket_name, s3_key, local_file_path)
    else:
        print("No objects found in the given S3 path.")
    return local_base_dir
