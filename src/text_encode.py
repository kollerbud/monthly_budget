'''Encoding text data'''
from abc import ABC, abstractmethod
from typing import Dict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
from data_prep import TrainCSVDataLoader, DataCleaner, DataLoader


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

    def city_encode(self) -> np.ndarray:
        'label encode city/state column'
        encode = LabelEncoder()
        cities = self.clean_data['City/State'].values
        city_encode = encode.fit_transform(cities.reshape(-1, 1).ravel())

        # save encoder
        joblib.dump(encode, 'saved_models/city_encoder.joblib')
        return city_encode

    def target_encode(self):
        '''encode target variable'''
        encode = LabelEncoder()
        category = self.clean_data['Category'].values
        category_encode = encode.fit_transform(category.reshape(-1, 1).ravel())

        # save encoder
        joblib.dump(encode, 'saved_models/category_encoder.joblib')
        return category_encode

    def transformed(self) -> np.ndarray:
        'combine encoded and unencoded columns'
        # (1207,635)
        desrp = self.descp_encode()
        # (1207, )
        cities = self.city_encode()
        # (1207, )
        target = self.target_encode()

        return {'feature_data': np.concatenate((desrp, cities[:, None]),
                                               axis=1),
                'target_data': target[: None]
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
        count_matrix = count_vec.fit_transform(self.clean_data['Description'])
        matrix_array = count_matrix.toarray()

        return matrix_array

    def city_encode(self) -> np.ndarray:
        'label encode city/state column'
        # load city encoder
        encode = joblib.load('saved_models/city_encoder.joblib')
        cities = self.clean_data['City/State'].values
        city_encode = encode.fit_transform(cities.reshape(-1, 1).ravel())

        return city_encode

    def target_encode(self) -> np.ndarray:
        '''encode target variable'''
        encode = joblib.load('saved_models/category_encoder.joblib')
        category = self.clean_data['Category'].values
        category_encode = encode.fit_transform(category.reshape(-1, 1).ravel())

        return category_encode

    def transformed(self) -> np.ndarray:
        'combine encoded and unencoded columns'
        # (1207,635)
        desrp = self.descp_encode()
        # (1207, )
        cities = self.city_encode()
        # (1207, )
        target = self.target_encode()

        # output feature dataset and target dataset
        return {'feature_data': np.concatenate((desrp, cities[:, None]),
                                               axis=1),
                'target_data': target[: None]
                }


class ProcessedData:
    '''combined pre-modeling steps together'''
    def __init__(self, loader: DataLoader,
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
    loader = TrainCSVDataLoader(file_path='raw_data/activity.csv')
    print(ProcessedData(loader=loader,
                        cleaner=DataCleaner,
                        encoder=LoadEncoder).run())
if __name__ == '__main__':
    run()
