'''Encoding text data'''
from typing import Type, TypedDict
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from data_upload import LoadFromBucket
from data_loader import DataCleaner


class MakeDataEncoder:
    '''encode and transform the text data from cleaned data'''
    feature_encoder = None
    target_encoder = None

    def __init__(self,
                 cleaned_data: Type[DataCleaner] = None,
                 ) -> None:
        self.clean_data = cleaned_data

    def descp_encode(self) -> np.ndarray:
        '''encode text columns'''
        # CountVectorizer
        self.feature_encoder = CountVectorizer(encoding='ISO-8859-1')
        count_matrix = self.feature_encoder.fit_transform(
                                self.clean_data['Description'])
        matrix_array = count_matrix.toarray()

        return matrix_array

    def columns_to_np(self):
        '''convert pandas columns to numpy array'''
        return self.clean_data['Amount'].to_numpy()

    def target_encode(self) -> np.ndarray:
        '''encode target variable'''
        self.target_encoder = LabelEncoder()
        category = self.clean_data['Category'].values
        category_encode = (self.target_encoder
                           .fit_transform(category.reshape(-1, 1).ravel())
                           )

        return category_encode

    def encoded_data(self) -> TypedDict:
        'combine encoded and unencoded columns'

        feature_data = np.concatenate(
                (self.descp_encode(), self.columns_to_np()[:, None]), axis=1)
        target_data = self.target_encode()
        return {'feature_data': feature_data,
                'target_data': target_data,
                'encoders': (self.feature_encoder,
                             self.target_encoder)
                }

    def save_encoders(self, path):
        'save encoders to model file'
        joblib.dump(self.feature_encoder,
                    f'{path}/desp_encoder.pkl')
        joblib.dump(self.target_encoder,
                    f'{path}/target_encoder.pkl')
        return f'encoders saved to {path}'


class LoadEncoderFromBucket:
    '''load trained encoders from GCP bucket'''
    feature_encoder = None
    target_encoder = None

    def __init__(self,
                 clean_data: Type[DataCleaner],
                 path: str) -> None:
        self.clean_data = clean_data
        self.path = path

    def load_vectorizer(self):
        'load saved vectorizer from gcp bucket'
        vec = LoadFromBucket().load_vector_from_bucket(
            bucket_name='monthly_budget_models',
            model_blob=self.path
        )

        return vec

    def descp_encode(self) -> np.ndarray:
        '''encode text columns'''
        # load CountVectorizer
        count_vec = joblib.load(self.load_vectorizer())[0]
        count_matrix = count_vec.transform(self.clean_data['Description'])
        matrix_array = count_matrix.toarray()

        return matrix_array

    def columns_to_np(self):
        '''convert pandas columns to numpy array'''
        return self.clean_data['Amount'].to_numpy()

    def target_encode(self) -> np.ndarray:
        '''encode target variable'''
        encode = joblib.load(self.load_vectorizer())[1]
        category = self.clean_data['Category'].values
        category_encode = encode.transform(category.reshape(-1, 1).ravel())

        return category_encode

    def encoded_data(self) -> np.ndarray:
        'combine encoded and unencoded columns'

        feature_data = np.concatenate(
                (self.descp_encode(), self.columns_to_np()[:, None]), axis=1)
        target_data = self.target_encode()
        # saved generated encoders
        return {'feature_data': feature_data,
                'target_data': target_data,
                }


class LoadEncoderFromLocal:
    '''load saved encoders from local path'''
    def __init__(self) -> None:
        pass

