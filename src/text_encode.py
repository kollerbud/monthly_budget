'''Encoding text data'''
from abc import ABC, abstractmethod
from typing import Dict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
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
    feature_encoder: CountVectorizer
    target_encoder: LabelEncoder

    def __init__(self,
                 clean_data: DataCleaner = None) -> None:

        self.clean_data = clean_data.cleaner_output()

    def descp_encode(self) -> np.ndarray:
        '''encode text columns'''
        # CountVectorizer
        self.feature_encoder = CountVectorizer(encoding='ISO-8859-1')
        count_matrix = self.feature_encoder.fit_transform(self.clean_data['Description'])
        matrix_array = count_matrix.toarray()

        return matrix_array

    def columns_to_np(self):
        '''convert pandas columns to numpy array'''
        return self.clean_data['Amount'].to_numpy()

    def target_encode(self) -> np.ndarray:
        '''encode target variable'''
        self.target_encoder = LabelEncoder()
        category = self.clean_data['Category'].values
        category_encode = self.target_encoder.fit_transform(category.reshape(-1, 1).ravel())
        
        return category_encode

    def transformed(self) -> np.ndarray:
        'combine encoded and unencoded columns'
        return {'feature_data': np.concatenate(
                (self.descp_encode(), self.columns_to_np()[:, None]), axis=1),
                'target_data': self.target_encode(),
                }

    def save_encoders(self, file_loc):
        pickle.dump(self.feature_encoder,
                    open(f'{file_loc}/desp_transformer.pkl', 'wb')
                    )
        pickle.dump(self.target_encoder,
                    open(f'{file_loc}/target_transformer.pkl', 'wb')
                    )


class LoadEncoder:
    '''load trained encoders'''
    feature_encoder: CountVectorizer
    target_encoder: LabelEncoder

    def __init__(self,
                 file_loc: str,
                 clean_data: DataCleaner,
                  ) -> None:
        self.clean_data = clean_data.cleaner_output()
        self.file_loc = file_loc

    def descp_encode(self) -> np.ndarray:
        '''encode text columns'''
        # load CountVectorizer
        count_vec = pickle.load(open(f'{self.file_loc}/desp_transformer.pkl', 'rb'))
        count_matrix = count_vec.transform(self.clean_data['Description'])
        matrix_array = count_matrix.toarray()

        return matrix_array

    def columns_to_np(self):
        '''convert pandas columns to numpy array'''
        return self.clean_data['Amount'].to_numpy()

    def target_encode(self) -> np.ndarray:
        '''encode target variable'''
        encode = pickle.load(open(f'{self.file_loc}/target_transformer.pkl', 'rb'))
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
    def __init__(self, 
                 loader,
                 cleaner: DataCleaner,
                 encoder: Encoder
                 ) -> None:
        self.loader = loader
        self.cleaner = cleaner
        self.encoder = encoder

    def run(self) -> Dict[str, np.ndarray]:
        'output data ready for modeling'
        load_output = self.loader
        clean_ouput = self.cleaner(load_output)

        return self.encoder(clean_ouput).transformed()



if __name__ == '__main__':
    def run():
        """
        loader = TrainCSVDataLoader(file_path='raw_data/all_year.csv',
                                    use_cols= ['Description', 'Amount', 'City/State',
                                                'Zip Code', 'Category'])
        print(ProcessedData(loader=loader,
                            cleaner=DataCleaner,
                            encoder=MakeEncoder).run())
        """
        loader = TrainCSVDataLoader(file_path='raw_data/all_year.csv',
                                    use_cols= ['Description', 'Amount', 'City/State',
                                                'Zip Code', 'Category'])
        cleaner = DataCleaner(loader)
        i = LoadEncoder(file_loc='saved_models', clean_data=cleaner)
        print(i.transformed())

        
    run()
