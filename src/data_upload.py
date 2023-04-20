from google.cloud import storage
from google.oauth2 import service_account
from dotenv import load_dotenv
import os

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
                     file_type: str, bucket_name: str):
        
        bucket = self.storage_client.get_bucket(bucket_name)
        print(bucket)
        #_files = [filename for filename in path if filename.endswith(f'{file_type}')]
        
        #for file in _files:
        #    blob = bucket.blob(file)
        #    blob.upload_from_filename(file)
            
        #    return blob.public_url
        
        


if __name__ == '__main__':
    
    UploadToBucket().upload_files(path='saved_models', file_type='.pkl', bucket_name='monthly_budget_models')