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


class DataCleaner:
    '''data processing steps for data from DataLoader'''

    def __init__(self, data_loader: DataLoader) -> None:
        self.data = data_loader.copy()

    def text_process(self) -> pd.DataFrame:
        '''text processes'''
        # remove all numbers from description, and punction symbols
        for col in ['Description', 'City/State']:
            # convert to lower case
            self.data[col] = self.data[col].str.lower()
            # text string clean up
            self.data[col] = (
                self.data[col]
                .str.replace(r'\d+', '', regex=True)
                .str.replace('[^\w\s]', '', regex=True)
                .str.replace('\n', ' ', regex=True)
                )
            # remove leading and trailing white spaces
            self.data[col] = self.data[col].str.strip()

        # remove excess white space between words
        self.data['Description'] = (
            self.data['Description'].str.replace(r'\s+', ' ', regex=True)
            )

        # remove city and state from string
        # by referencing "City/State" column
        self.data['Description'] = [
            x.replace(str(y), '') for x, y in
            zip(self.data['Description'],
                self.data['City/State'])
            ]

        return self.data

    def remove_payment(self) -> pd.DataFrame:
        'remove payment rows where I paid the credit amount'
        payment_mask = self.data['Category'].isin(['Payment'])
        self.data = self.data[~payment_mask]

    def cleaner_output(self) -> pd.DataFrame:
        'run all cleaning processes and output the clean data'
        # run all data cleaning steps
        self.text_process()
        self.remove_payment()
        # last check for data modification
        if self.data is None:
            raise ValueError('No data has been processed')
        return self.data

def run():

    loader = TrainCSVDataLoader(file_path='raw_data/activity.csv').loader_output()
    text_processor = DataCleaner(data_loader=loader).cleaner_output()
    print(text_processor.sample(10))

if __name__ == '__main__':
    run()
