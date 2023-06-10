import os
from io import BytesIO
from google.cloud import storage
from google.oauth2 import service_account
from dotenv import load_dotenv


load_dotenv()


class MLFlowToBucket:
    '''specific to local mlflow development'''

    def __init__(self) -> None:
        if os.getenv('private_key') is None:
            self.storage_client = storage.Client()

        self.storage_client = self.client_detail()

    def client_detail(self):
        secrets = {}
        for i in ['type', 'project_id', 'private_key_id',
                  'client_email', 'client_id', 'auth_uri', 'token_uri',
                  'auth_provider_x509_cert_url', 'client_x509_cert_url']:
            secrets[i] = os.getenv(i)
            # need to clean up private_key
        secrets['private_key'] = os.getenv('private_key').replace('\\n', '\n')
        cred = service_account.Credentials.from_service_account_info(secrets)
        # role permission issue
        return storage.Client(credentials=cred)

    def upload_files(self, path: str,
                     bucket_name: str
                     ):
        bucket = self.storage_client.get_bucket(bucket_name)
        # full path
        model_path = path + '/artifacts/budget'
        model_files = [filename for filename in
                       os.listdir(model_path)]

        for file in model_files:
            # blob is destination folder structure
            blob = bucket.blob(model_path+'/' + file)
            blob.upload_from_filename(model_path+'/'+file)

        vector_path = path + '/artifacts/vectorizer'
        vector_files = [filename for filename in
                        os.listdir(vector_path)]

        for file in vector_files:
            # blob is destination folder structure
            blob = bucket.blob(vector_path+'/' + file)
            blob.upload_from_filename(vector_path+'/'+file)


class LoadFromBucket:
    '''load model from Bucket'''
    def __init__(self) -> None:
        if os.getenv('private_key') is None:
            self.storage_client = storage.Client()
        self.storage_client = self.client_detail()

    def client_detail(self):
        secrets = {}
        for i in ['type', 'project_id', 'private_key_id',
                  'client_email', 'client_id', 'auth_uri', 'token_uri',
                  'auth_provider_x509_cert_url', 'client_x509_cert_url']:
            secrets[i] = os.getenv(i)
            # need to clean up private_key
        secrets['private_key'] = os.getenv('private_key').replace('\\n', '\n')
        cred = service_account.Credentials.from_service_account_info(secrets)
        # role permission issue
        return storage.Client(credentials=cred)

    def load_vector_from_bucket(self, bucket_name, model_blob):
        bucket = self.storage_client.get_bucket(bucket_name)

        blob_path = model_blob + '/artifacts/vectorizer/model.pkl'

        blob = bucket.blob(blob_path)

        a = blob.download_as_bytes()

        return BytesIO(a)

    def load_model_from_bucket(self, bucket_name, model_blob):
        bucket = self.storage_client.get_bucket(bucket_name)

        blob_path = model_blob + '/artifacts/budget/model.pkl'

        blob = bucket.blob(blob_path)

        a = blob.download_as_bytes()

        return BytesIO(a)


if __name__ == '__main__':
    load_dotenv()
    LoadFromBucket().client_detail()
