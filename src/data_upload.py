from google.cloud import storage
from google.oauth2 import service_account
from dotenv import load_dotenv
import os
from io import BytesIO


load_dotenv()


class UploadToBucket:

    def __init__(self) -> None:
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
                     bucket_name: str,
                     #dest_blob: str
                     ):
        # bucket
        bucket = self.storage_client.get_bucket(bucket_name)
        _files = [filename for filename in os.listdir(path)]

        for file in _files:
            # blob is destination folder structure
            blob = bucket.blob(path+'/'+file)
            blob.upload_from_filename(path+'/'+file)

            print(blob.public_url)


class LoadFromBucket:
    '''load model from Bucket'''
    def __init__(self) -> None:
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

    def load_file_from_bucket(self, bucket_name, model_blob):
        bucket = self.storage_client.get_bucket(bucket_name)
        blob = bucket.blob(model_blob)
        
        a = blob.download_as_bytes()

        return BytesIO(a)



if __name__ == '__main__':

    x = LoadFromBucket().load_file_from_bucket(bucket_name='monthly_budget_models',
                        model_blob='mlruns/1/23c98085d10d4e8897d48c352d3a71df/artifacts/budget/model.pkl'
                        )
    import joblib
    
    print(joblib.load(x))
    
