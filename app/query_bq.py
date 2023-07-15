from typing import List
from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core.client_options import ClientOptions
from dotenv import load_dotenv
import os


load_dotenv()


class RecentSpendsFromBQ:

    def __init__(self) -> None:

        if os.getenv('private_key') is None:
            self.storage_client = bigquery.Client()

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
        # permission issue with Bigquery using sheets as backend
        # detail-- https://github.com/googleapis/google-auth-library-python/issues/1204
        options = ClientOptions(
                    scopes=['https://www.googleapis.com/auth/drive',
                            'https://www.googleapis.com/auth/cloud-platform']
                    )

        return bigquery.Client(credentials=cred,
                               client_options=options)

    def build_query(self,
                    cols: List[str] = None,
                    num_months: int = 1
                    ):
        # cannot parameterize select claus
        join_cols = ', '.join(cols)

        query_string = f'''SELECT {join_cols}
                        FROM `rolling_statements.spend_statements2`
                        WHERE Date >= DATE_SUB(CURRENT_DATE(), INTERVAL @num MONTH)
                        ORDER BY Date DESC
                        ;
                        '''
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter('num', 'INT64', num_months)
            ]
        )
        # run query
        query_job = (
            self.client_detail().query(
                        query=query_string,
                        job_config=job_config
                        ).to_dataframe()
        )
        return query_job

if __name__ == '__main__':

    print(RecentSpendsFromBQ().build_query(
                cols=['Date', 'Description', 'City_State', 'Amount',
                      'Zip_Code', 'Category'],
                num_months=1
                )
          )
