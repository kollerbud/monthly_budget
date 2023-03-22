'''Loading and preprocessing raw data'''
from typing import List
import pandas as pd


class TrainCSVDataLoader:
    '''read a CSV file and output dataframe'''

    def __init__(self,
                 file_path: str,
                 use_cols: List = None) -> None:
        self.path = file_path
        self.to_dataframe()
        if use_cols is not None:
            self.cols = use_cols

    def to_dataframe(self) -> pd.DataFrame:
        'read the csv file to a dataframe'
        return pd.read_csv(self.path, encoding='ISO-8859-1')

    def select_cols(self) -> pd.DataFrame:
        '''select the columns to use from dataframe'''
        return self.to_dataframe().loc[:, self.cols]

    def loader_output(self) -> pd.DataFrame:
        '''return the trimmed dataframe'''
        return self.select_cols()


class MonthlyCSVDataLoader:
    '''read table from Bigquery'''
    def __init__(self) -> None:
        pass


class DataCleaner:
    '''data processing steps for data from DataLoader'''

    def __init__(self, data_loader) -> None:
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

    def clean_category(self):
        '''clean up categories'''
        # blank line is the payment, should remove
        self.data.dropna(subset=['Category'], inplace=True)
        self.data['Category'] = self.data['Category'].str.split('-').str[0]

    def remove_payment(self) -> None:
        'remove payment rows where I paid the credit amount'
        payment_mask = self.data['Category'].isin(['Payment'])
        self.data = self.data[~payment_mask]

    def remove_cities(self) -> None:
        'remove City/State column before encoding'
        del self.data['City/State']

    def cleaner_output(self) -> pd.DataFrame:
        'run all cleaning processes and output the clean data'
        # run all data cleaning steps
        self.text_process()
        self.clean_category()
        self.remove_payment()
        self.remove_cities()
        # last check for data modification
        if self.data is None:
            raise ValueError('No data has been processed')
        return self.data

def run():

    loader = TrainCSVDataLoader(file_path='raw_data/all_year.csv',
                                use_cols= ['Description', 'Amount', 'City/State',
                                        'Zip Code', 'Category']).loader_output()
    text_processor = DataCleaner(data_loader=loader).cleaner_output()
    print(text_processor.sample(10))

if __name__ == '__main__':
    run()
