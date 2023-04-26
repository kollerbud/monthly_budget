'''Encoding text data'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from data_upload import LoadFromBucket
import numpy as np
import joblib
from data_loader import DataCleaner


class MakeEncoder:
    '''encode and transform the text data from cleaned data'''
    def __init__(self,
                 clean_data: DataCleaner = None,
                 encoder_path: str = None) -> None:
        if encoder_path is not None:
            self.path = encoder_path
        self.clean_data = clean_data
        self.feature_encoder = None
        self.target_encoder = None

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

    def encoded_data(self) -> np.ndarray:
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


class Data_Processor:

    _encoder = None

    def __init__(self, loader, cleaner, encoder, encoder_path=None) -> None:
        self.loader = loader
        self.cleaner = cleaner
        self.encoder = encoder
        self.path = encoder_path

    def run(self):
        load_data = self.loader.loader_output()

        clean_data = self.cleaner(data_loader=load_data).cleaner_output()

        encoder = self.encoder(clean_data=clean_data, encoder_path=self.path)

        # Encoder class not running until this step
        encoded_data = encoder.encoded_data()
        self._encoder = encoder

        return encoded_data

    def save_encoder(self):
        self.run()
        self._encoder.save_encoders(path=self.path)


class LoadEncoderFromBucket:
    '''load trained encoders'''

    def __init__(self,
                 encoder_path: str,
                 clean_data: DataCleaner,
                 ) -> None:
        self.path = encoder_path
        self.clean_data = clean_data
        self.feature_encoder = None
        self.target_encoder = None

    def load_vectorizer(self):
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

    def __init__(self) -> None:
        pass
