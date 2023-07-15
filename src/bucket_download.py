import os
from typing import Type
from io import BytesIO
from google.cloud import storage
from google.oauth2 import service_account
from dotenv import load_dotenv
from _auth import credentials


load_dotenv()


class LoadFromBucket:
    '''load model from Bucket'''
    def __init__(self) -> None:
        if os.getenv('private_key') is None:
            self.storage_client = storage.Client()
        self.storage_client = self.client_detail()

    def client_detail(self):
        # role permission issue
        return storage.Client(credentials=credentials())

    def load_from_uri(self,
                      file_bucket: str = None,
                      file_name: str = None) -> 'BytesIO':
        file_uri = file_bucket + '/' + file_name
        uri_parts = file_uri.split('//')[1].split('/')
        bucket_name = uri_parts[0]
        object_name = '/'.join(uri_parts[1:])

        bucket = self.storage_client.bucket(bucket_name=bucket_name)
        blob = bucket.blob(object_name)

        a = blob.download_as_bytes()

        return BytesIO(a)
