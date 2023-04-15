'''Encoding text data'''
from typing import Dict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
from data_prep import TrainCSVDataLoader, DataCleaner


class MakeEncoder:
    '''encode and transform the text data from cleaned data'''
    def __init__(self,
                 clean_data: DataCleaner = None,
                 encoder_path: str = None) -> None:
        self.path = encoder_path
        self.clean_data = clean_data
        self.feature_encoder = None
        self.target_encoder = None

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

    def save_encoders(self):
        'save encoders to model file'
        pickle.dump(self.feature_encoder,
                    open(f'{self.path}/desp_encoder.pkl', 'wb')
                    )
        pickle.dump(self.target_encoder,
                    open(f'{self.path}/target_encoder.pkl', 'wb')
                    )
        return f'encoders saved to {self.path}'

    def encoded(self) -> np.ndarray:
        'combine encoded and unencoded columns'
        feature_data = np.concatenate(
                (self.descp_encode(), self.columns_to_np()[:, None]), axis=1)
        target_data = self.target_encode()
        # saved generated encoders
        self.save_encoders()
        return {'feature_data': feature_data,
                'target_data': target_data,
                }


class LoadEncoder:
    '''load trained encoders'''

    def __init__(self,
                 encoder_path: str,
                 clean_data: DataCleaner,
                 ) -> None:
        self.clean_data = clean_data
        self.path = encoder_path
        self.feature_encoder = None
        self.target_encoder = None        

    def descp_encode(self) -> np.ndarray:
        '''encode text columns'''
        # load CountVectorizer
        count_vec = pickle.load(open(f'{self.path}/desp_encoder.pkl', 'rb'))
        count_matrix = count_vec.transform(self.clean_data['Description'])
        matrix_array = count_matrix.toarray()

        return matrix_array

    def columns_to_np(self):
        '''convert pandas columns to numpy array'''
        return self.clean_data['Amount'].to_numpy()

    def target_encode(self) -> np.ndarray:
        '''encode target variable'''
        encode = pickle.load(open(f'{self.path}/target_encoder.pkl', 'rb'))
        category = self.clean_data['Category'].values
        category_encode = encode.transform(category.reshape(-1, 1).ravel())

        return category_encode

    def encoded(self) -> np.ndarray:
        'combine encoded and unencoded columns'
        feature_data = np.concatenate(
                (self.descp_encode(), self.columns_to_np()[:, None]), axis=1)
        target_data = self.target_encode()
        # saved generated encoders
        return {'feature_data': feature_data,
                'target_data': target_data,
                }



class ProcessedData:
    '''combined pre-modeling steps together'''
    def __init__(self,
                 loader,
                 cleaner,
                 encoder,
                 encoder_path: str = None
                 ) -> None:
        self.loader = loader
        self.cleaner = cleaner
        self.encoder = encoder
        self.path = encoder_path
    
    def run(self) -> Dict[str, np.ndarray]:
        'output data ready for modeling'
        load_output = self.loader.loader_output()
        clean_output = self.cleaner(load_output).cleaner_output()
        encoded = self.encoder(encoder_path=self.path,
                               clean_data=clean_output)

        return encoded.encoded()


if __name__ == '__main__':
    def run():
        loader = TrainCSVDataLoader(file_path='raw_data/all_year.csv',
                                    use_cols= ['Description', 'Amount', 'City/State',
                                                'Zip Code', 'Category'])
        x = ProcessedData(loader=loader,
                          cleaner=DataCleaner,
                          encoder=LoadEncoder,
                          save_model=False,
                          encoder_path='saved_models').run()
        print(x)

    run()
