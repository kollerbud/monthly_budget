'''Loading and preprocessing raw data'''
from typing import List, Type
import pandas as pd
from google.cloud import bigquery
from google.api_core.client_options import ClientOptions
from dotenv import load_dotenv
from _auth import credentials


load_dotenv()


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
                 months: int = 12) -> None:

        self.client = self._client_detail()
        self._data = None
        self.months = months

    @staticmethod
    def _client_detail():
        # permission issue with Bigquery using sheets as backend
        # detail-- https://github.com/googleapis/google-auth-library-python/issues/1204
        options = ClientOptions(
                    scopes=['https://www.googleapis.com/auth/drive',
                            'https://www.googleapis.com/auth/cloud-platform']
                    )
        return bigquery.Client(credentials=credentials(),
                               client_options=options)

    def query_table(self) -> pd.DataFrame:
        query_string = '''
            SELECT Description, Amount, City_State, Zip_Code, Category
            FROM `rolling_statements.spend_statements2`
            WHERE Date >= DATE_SUB(CURRENT_DATE(), INTERVAL @num MONTH)
            ORDER BY Date DESC
            ;
        '''
        job_configs = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter('num', 'INT64', self.months)
            ]
        )
        query_job = (
            self.client.query(
                query=query_string,
                job_config=job_configs,
                ).to_dataframe()
        )
        return query_job

    def loader_output(self) -> pd.DataFrame:
        'run all loading steps and perform data checks'
        self._data = self.query_table()
        self._data.rename(columns={'City_State': 'City/State'}, inplace=True)
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


if __name__ == '__main__':
    loader = (BigqueryDataLoader().loader_output())
    print(loader)
    print(DataCleaner(data_loader=loader).cleaner_output())
