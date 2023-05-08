'''Loading and preprocessing raw data'''
from typing import List, Type
import os
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account


class CSVDataLoader:
    '''read a CSV file, return dataframe'''

    def __init__(self,
                 file_path: str,
                 cols: List[str] = None) -> None:
        self.path = file_path
        self.cols = cols
        self._data = None

    def _req_columns_check(self):
        '''check if all required columns are in dataframe'''
        req_cols = ['Description', 'Amount', 'City/State',
                    'Zip Code', 'Category']

        if all(elem in self._data.columns for elem in req_cols):
            return True
        else:
            return False

    def to_dataframe(self) -> pd.DataFrame:
        'read the csv file to a dataframe'
        self._data = pd.read_csv(self.path, encoding='ISO-8859-1')

    def select_cols(self) -> pd.DataFrame:
        '''select the columns to use from dataframe'''
        self.to_dataframe()
        self._data = self._data.loc[:, self.cols]

    def loader_output(self) -> pd.DataFrame:
        'run all loading steps and perform data checks'
        self.to_dataframe()
        self.select_cols()

        if self._data is None:
            raise ValueError('data is none')

        if self._req_columns_check() is False:
            raise ValueError('not all required columns are loaded')
        return self._data


class JsonDataLoader:
    '''read json data and return dataframe'''

    def __init__(self,
                 file_path: str,
                 cols: List[str] = None) -> None:
        self.path = file_path
        self.cols = cols
        self._data = None

    def _req_columns_check(self):
        '''check if all required columns are in dataframe'''
        req_cols = ['Description', 'Amount', 'City/State',
                    'Zip Code', 'Category']

        if all(elem in self._data.columns for elem in req_cols):
            return True
        else:
            return False

    def to_dataframe(self) -> pd.DataFrame:
        'read the csv file to a dataframe'
        self._data = pd.read_json(self.path, encoding='ISO-8859-1')

    def select_cols(self) -> pd.DataFrame:
        '''select the columns to use from dataframe'''
        return self._data.loc[:, self.cols]

    def loader_output(self) -> pd.DataFrame:
        'run all loading steps and perform data checks'
        self.to_dataframe()
        self.select_cols()

        if self._data is None:
            raise ValueError('data is none')

        if self._req_columns_check() is False:
            raise ValueError('not all required columns are loaded')
        return self._data


class BigqueryDataLoader:
    '''read json data and return dataframe'''

    def __init__(self,
                 file_path: str,
                 cols: List[str] = None) -> None:
        self._client_detail()

    def _client_detail(self):
        secrets = {}
        for i in ['type', 'project_id', 'private_key_id',
                  'client_email', 'client_id', 'auth_uri', 'token_uri',
                  'auth_provider_x509_cert_url', 'client_x509_cert_url']:
            secrets[i] = os.getenv(i)
            # need to clean up private_key
        secrets['private_key'] = os.getenv('private_key').replace('\\n', '\n')
        cred = service_account.Credentials.from_service_account_info(secrets)
        return bigquery.Client(credentials=cred)
    
    def build_query(self):
        '''file path should be the table'''
        query_string = '''
            SELECT @columns
            FROM @bq_path
            WHERE 
        '''

    def _req_columns_check(self):
        '''check if all required columns are in dataframe'''
        req_cols = ['Description', 'Amount', 'City/State',
                    'Zip Code', 'Category']

        if all(elem in self._data.columns for elem in req_cols):
            return True
        else:
            return False

    def to_dataframe(self) -> pd.DataFrame:
        'read the csv file to a dataframe'
        self._data = pd.read_json(self.path, encoding='ISO-8859-1')

    def select_cols(self) -> pd.DataFrame:
        '''select the columns to use from dataframe'''
        return self._data.loc[:, self.cols]

    def loader_output(self) -> pd.DataFrame:
        'run all loading steps and perform data checks'
        self.to_dataframe()
        self.select_cols()

        if self._data is None:
            raise ValueError('data is none')

        if self._req_columns_check() is False:
            raise ValueError('not all required columns are loaded')
        return self._data

class DataCleaner:
    '''data processing steps for data from DataLoader'''

    def __init__(self, data_loader: Type[CSVDataLoader]) -> None:
        self.data = data_loader

    def text_process(self) -> pd.DataFrame:
        '''text processes'''

        for col in ['Description', 'City/State']:
            self.data[col] = self.data[col].str.lower()
            self.data[col] = (
                self.data[col]
                .str.replace(r'\d+', '', regex=True)
                .str.replace('[^\w\s]', '', regex=True)
                .str.replace('\n', ' ', regex=True)
                )
            self.data[col] = self.data[col].str.strip()

        self.data['Description'] = (
            self.data['Description'].str.replace(r'\s+', ' ', regex=True)
            )

        self.data['Description'] = [
            x.replace(str(y), '') for x, y in
            zip(self.data['Description'],
                self.data['City/State'])
            ]
        return self.data

    def clean_category(self) -> None:
        '''drop empty category'''
        self.data.dropna(subset=['Category'], inplace=True)
        self.data['Category'] = self.data['Category'].str.split('-').str[0]

    def remove_cities(self) -> None:
        'remove City/State column before encoding'
        del self.data['City/State']

    def cleaner_output(self) -> pd.DataFrame:
        'run all cleaning processes and output the clean data'
        # run all data cleaning steps
        self.text_process()
        self.clean_category()
        self.remove_cities()
        # last check for data modification
        if self.data is None:
            raise ValueError('No data has been processed')
        return self.data
