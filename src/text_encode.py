'''Encoding text data'''
from abc import ABC, abstractmethod
from typing import Dict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
from data_prep import TrainCSVDataLoader, DataCleaner


class Encoder(ABC):
    '''encode text columns'''
    @abstractmethod
    def descp_encode(self) -> np.ndarray:
        '''encode description column'''

    @abstractmethod
    def target_encode(self) -> np.ndarray:
        '''encode target variable'''

    @abstractmethod
    def transformed(self) -> np.ndarray:
        '''output the transformed data'''


class MakeEncoder(Encoder):
    '''encode and transform the text data from cleaned data'''

    def __init__(self,
                 clean_data: DataCleaner = None) -> None:

        self.clean_data = clean_data

    def descp_encode(self) -> np.ndarray:
        '''encode text columns'''
        # CountVectorizer
        count_vec = CountVectorizer(encoding='ISO-8859-1')
        count_matrix = count_vec.fit_transform(self.clean_data['Description'])
        matrix_array = count_matrix.toarray()

        # save encoder
        joblib.dump(count_vec, 'saved_models/vectorizer.joblib')
        return matrix_array

    def columns_to_np(self):
        '''convert pandas columns to numpy array'''
        return self.clean_data['Amount'].to_numpy()

    def target_encode(self) -> np.ndarray:
        '''encode target variable'''
        encode = LabelEncoder()
        category = self.clean_data['Category'].values
        category_encode = encode.fit_transform(category.reshape(-1, 1).ravel())

        # save encoder
        joblib.dump(encode, 'saved_models/category_encoder.joblib')
        return category_encode

    def transformed(self) -> np.ndarray:
        'combine encoded and unencoded columns'
        return {'feature_data': np.concatenate(
                (self.descp_encode(), self.columns_to_np()[:, None]), axis=1),
                'target_data': self.target_encode(),
                }


class LoadEncoder:
    '''load trained encoders'''

    def __init__(self,
                 clean_data: DataCleaner = None) -> None:
        self.clean_data = clean_data

    def descp_encode(self) -> np.ndarray:
        '''encode text columns'''
        # load CountVectorizer
        count_vec = joblib.load('saved_models/vectorizer.joblib')
        count_matrix = count_vec.transform(self.clean_data['Description'])
        matrix_array = count_matrix.toarray()

        return matrix_array
    
    def columns_to_np(self):
        '''convert pandas columns to numpy array'''
        return self.clean_data['Amount'].to_numpy()

    def target_encode(self) -> np.ndarray:
        '''encode target variable'''
        encode = joblib.load('saved_models/category_encoder.joblib')
        category = self.clean_data['Category'].values
        category_encode = encode.transform(category.reshape(-1, 1).ravel())

        return category_encode

    def transformed(self) -> np.ndarray:
        'combine encoded and unencoded columns'
        # output feature dataset and target dataset
        return {'feature_data': np.concatenate(
                (self.descp_encode(), self.columns_to_np()[:, None]), axis=1),
                'target_data': self.target_encode(),
                }


class ProcessedData:
    '''combined pre-modeling steps together'''
    def __init__(self, loader,
                 cleaner: DataCleaner,
                 encoder: Encoder
                 ) -> None:
        self.loader = loader
        self.cleaner = cleaner
        self.encoder = encoder

    def run(self) -> Dict[str, np.ndarray]:
        'output data ready for modeling'
        load_output = self.loader.loader_output()
        clean_ouput = self.cleaner(load_output).cleaner_output()

        return self.encoder(clean_ouput).transformed()


def run():
    loader = TrainCSVDataLoader(file_path='raw_data/activity.csv',
                                use_cols= ['Description', 'Amount', 'City/State',
                                            'Zip Code', 'Category'])
    print(ProcessedData(loader=loader,
                        cleaner=DataCleaner,
                        encoder=MakeEncoder).run())
if __name__ == '__main__':
    run()
