'''Loading and preprocessing raw data'''
from typing import List
import pandas as pd


class CSVDataLoader:
    '''read a CSV file, return dataframe'''

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


class JsonDataLoader:
    '''read json data and return dataframe'''

    def __init__(self,
                 file_path: str,
                 use_cols: List = None) -> None:
        self.path = file_path
        self.to_dataframe()
        if use_cols is not None:
            self.cols = use_cols

    def to_dataframe(self) -> pd.DataFrame:
        'read the csv file to a dataframe'
        return pd.read_json(self.path)

    def select_cols(self) -> pd.DataFrame:
        '''select the columns to use from dataframe'''
        return self.to_dataframe().loc[:, self.cols]

    def loader_output(self) -> pd.DataFrame:
        '''return the trimmed dataframe'''
        return self.select_cols()


class DataCleaner:
    '''data processing steps for data from DataLoader'''

    def __init__(self, data_loader) -> None:
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
    def run():
        loader = CSVDataLoader(file_path='raw_data/activity_feb.csv',
                                    use_cols= ['Description', 'Amount', 'City/State',
                                            'Zip Code', 'Category']).loader_output()

        text_processor = DataCleaner(data_loader=loader).cleaner_output()

    run()
