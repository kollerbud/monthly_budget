'''Loading and preprocessing raw data'''
from abc import ABC, abstractmethod
import pandas as pd


class DataLoader(ABC):
    '''class for loading data'''
    @abstractmethod
    def to_dataframe(self):
        '''load data into dataframe'''

    @abstractmethod
    def loader_output(self):
        '''return the dataframe'''


class TrainCSVDataLoader(DataLoader):
    '''read a CSV file and output dataframe'''

    def __init__(self, file_path: str) -> None:
        self.path = file_path
        self.to_dataframe()

    def to_dataframe(self) -> DataLoader:
        'read the csv file to a dataframe'
        return pd.read_csv(self.path, encoding='ISO-8859-1')

    def select_cols(self) -> pd.DataFrame:
        '''select the columns to use from dataframe'''
        return self.to_dataframe().loc[:,
                                       ['Description', 'Amount', 'City/State',
                                        'Zip Code', 'Category']]

    def loader_output(self) -> pd.DataFrame:
        '''return the trimmed dataframe'''
        return self.select_cols()


class BigqueryDataLoader(DataLoader):
    '''read table from Bigquery'''
    def __init__(self) -> None:
        pass


class DataCleaning:
    '''data processing steps for data from DataLoader'''
    _data = None

    def __init__(self, data_loader: DataLoader) -> None:
        self.data = data_loader

    def text_process(self) -> pd.DataFrame:
        '''text processes'''
        # make a copy of raw data
        self._data = self.data.loader_output().copy()

        # remove all numbers from description, and punction symbols
        for col in ['Description', 'City/State']:
            # convert to lower case
            self._data[col] = self._data[col].str.lower()
            # text string clean up
            self._data[col] = (
                self._data[col]
                .str.replace(r'\d+', '', regex=True)
                .str.replace('[^\w\s]', '', regex=True)
                .str.replace('\n', ' ', regex=True)
                )
            # remove leading and trailing white spaces
            self._data[col] = self._data[col].str.strip()

        # remove excess white space between words
        self._data['Description'] = (
            self._data['Description'].str.replace(r'\s+', ' ', regex=True)
            )

        # remove city and state from string
        # by referencing "City/State" column
        self._data['Description'] = [
            x.replace(str(y), '') for x, y in
            zip(self._data['Description'],
                self._data['City/State'])
            ]

        return self._data

    def remove_payment(self):
        'remove payment rows where I paid the credit amount'
        payment_mask = self._data['Category'].isin(['Payment'])
        self._data = self._data[~payment_mask]

    def clean_output(self):
        'run all cleaning processes and output the clean data'
        # run all data cleaning steps
        self.text_process()
        self.remove_payment()
        # last check for _data modification
        if self._data is None:
            raise ValueError('No data has been processed')
        return self._data

def run():

    loader = TrainCSVDataLoader(file_path='rawData/activity.csv')
    text_processor = DataCleaning(data_loader=loader).clean_output()
    print(text_processor.sample(10))


if __name__ == '__main__':
    run()
