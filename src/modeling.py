'Modeling and metrics here'
from typing import Type
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from encoding import MakeDataEncoder
from bucket_download import LoadFromBucket


class TrainModel:
    '''Use prepared data to train a model'''
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    def __init__(self,
                 encoded_data: Type[MakeDataEncoder],
                 model=None) -> None:

        self._data = encoded_data
        self.model = model

    def over_sample(self):
        '''balance income data'''
        overSampler = RandomOverSampler(random_state=21)

        (x_resample, y_resample) = (overSampler.fit_resample(
                                        self._data['feature_data'],
                                        self._data['target_data']
                                        )
                                    )

        return x_resample, y_resample

    def split(self) -> None:
        # resample to get balanced data
        x, y = self.over_sample()
        'split data into train and test'
        (self.x_train, self.x_test,
         self.y_train, self.y_test) = (
            train_test_split(x, y,
                             train_size=0.75,
                             stratify=y,
                             random_state=42)
            )

    def train_model(self):
        '''train model'''
        self.split()
        self.model.fit(self.x_train, self.y_train)
        print(f'model accuracy score is: {self.score()}')
        return self

    def score(self):
        'output model score'
        return self.model.score(self.x_test, self.y_test)

    def save_models(self, model_loc: str = None):
        'save current model'
        joblib.dump(self.model, f'{model_loc}/model.pkl')
        return f'model saved to "{model_loc}"'


class LoadModelFromBucket:
    '''Load and use existing models'''
    model = None
    target_encoder = None

    def __init__(self,
                 encoded_data: Type[MakeDataEncoder]=None
                 ) -> None:

        self._data = encoded_data

    def load_bucket_model(self,
                          bucket_name: str,
                          model_name: str):
        mod = LoadFromBucket().load_from_uri(
            file_bucket=bucket_name,
            file_name=model_name
        )
        self.model = joblib.load(mod)
        return self

    def load_target_encoder(
            self,
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

    def make_prediction(self) -> np.ndarray:
        'predict using model'

        return self.model.predict(self._data['feature_data'])

    def result(self) -> np.ndarray:
        # check model and encoder loading
        if self.model is None:
            raise ValueError('model is not loaded')
        if self.target_encoder is None:
            raise ValueError('encoder is not loaded')
        # translate predicted outputs to previous encoded names
        pred = self.make_prediction()
        decode = self.target_encoder.inverse_transform(pred)

        return decode
