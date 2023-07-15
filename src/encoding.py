'''Encoding text data'''
from typing import Type, TypedDict
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from bucket_download import LoadFromBucket
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
        category_encode = (
            self.target_encoder.fit_transform(category.reshape(-1, 1).ravel())
            )

        return category_encode

    def encode_data(self) -> TypedDict:
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
                 clean_data: Type[DataCleaner] = None,
                 ) -> None:
        self.clean_data = clean_data

    def load_vectorizer(self,
                        bucket_name: str,
                        encoder_name: str,
                        ):
        'load saved vectorizer from gcp bucket'
        vec = LoadFromBucket().load_from_uri(
            file_bucket=bucket_name,
            file_name=encoder_name
        )
        self.feature_encoder = joblib.load(vec)
        return self

    def load_target_encoder(self,
                            bucket_name: str,
                            encoder_name: str,
                            ):
        'load saved vectorizer from gcp bucket'
        vec = LoadFromBucket().load_from_uri(
            file_bucket=bucket_name,
            file_name=encoder_name
        )
        self.target_encoder = joblib.load(vec)
        return self

    def descp_encode(self) -> np.ndarray:
        '''encode text columns'''
        # load CountVectorizer
        if self.feature_encoder is None:
            raise ValueError('feature encoder is None')
        count_matrix = (
            self.feature_encoder.transform(self.clean_data['Description'])
            )
        matrix_array = count_matrix.toarray()

        return matrix_array

    def columns_to_np(self):
        '''convert pandas columns to numpy array'''
        return self.clean_data['Amount'].to_numpy()

    def target_encode(self) -> np.ndarray:
        '''encode target variable'''
        if self.target_encoder is None:
            raise ValueError('target encoder is None')
        category = self.clean_data['Category'].values
        category_encode = (
            self.target_encoder.transform(category.reshape(-1, 1).ravel())
            )

        return category_encode

    def encode_data(self) -> np.ndarray:
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

if __name__ == '__main__':
    None