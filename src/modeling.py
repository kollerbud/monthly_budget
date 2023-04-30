'Modeling and metrics here'
from typing import Type
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from encoding import MakeDataEncoder
from data_upload import LoadFromBucket


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

    def split(self) -> None:
        'split data into train and test'
        (self.x_train, self.x_test,
         self.y_train, self.y_test) = (
            train_test_split(self._data['feature_data'],
                             self._data['target_data'],
                             train_size=0.75,
                             stratify=self._data['target_data'],
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

    def __init__(self,
                 encoded_data: Type[MakeDataEncoder],
                 model_path=None) -> None:

        self._data = encoded_data
        self.model_path = model_path

    def load_bucket_model(self):
        mod = LoadFromBucket().load_model_from_bucket(
            bucket_name='monthly_budget_models',
            model_blob=self.model_path
        )
        return mod

    def load_model(self):
        'load existing model'
        self.model = joblib.load(self.load_bucket_model())

    def load_encoder(self):
        'load decoder to translate back predictions'
        decoder = LoadFromBucket().load_vector_from_bucket(
            bucket_name='monthly_budget_models',
            model_blob=self.model_path
        )
        return joblib.load(decoder)[1]

    def make_prediction(self) -> np.ndarray:
        'predict using model'
        self.load_model()
        return self.model.predict(self._data['feature_data'])

    def result(self) -> np.ndarray:
        'translate predicted outputs to previous encoded names'
        pred = self.make_prediction()
        encoder = self.load_encoder()
        decode = encoder.inverse_transform(pred)

        return decode
