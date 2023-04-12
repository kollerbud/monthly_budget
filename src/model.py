'Modeling and metrics here'
import joblib
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from text_encode import ProcessedData


class TrainModel:
    '''Use prepared data to train a model'''

    x_train = None
    x_test = None
    y_train = None
    y_test = None
    model = None

    def __init__(self,
                 prep_data: ProcessedData) -> None:

        self.prep_data = prep_data

    def split(self, train_size: float = 0.75) -> None:
        'split data into train and test'
        (self.x_train, self.x_test,
         self.y_train, self.y_test) = (
            train_test_split(self.prep_data['feature_data'],
                             self.prep_data['target_data'],
                             train_size=train_size,
                             random_state=42)
            )

        return self

    def train_model(self, model=None):
        '''train model'''
        self.model = model.fit(self.x_train, self.y_train)
        print(f'model accuracy score is: {self.score()}')
        return self

    def score(self):
        'output model score'
        return self.model.score(self.x_test, self.y_test)


class UseModel:
    '''Load and use existing models'''

    model = None

    def __init__(self,
                 prep_data: ProcessedData) -> None:
        self.prep_data = prep_data

    def load_model(self, model_path):
        'load existing model'
        self.model = pickle.load(open(model_path, 'rb'))
        return self

    def load_decoder(self, file_path) -> pickle:
        'load decoder to translate back predictions'
        return joblib.load(filename=file_path)

    def make_prediction(self) -> np.ndarray:
        'predict using model'
        return self.model.predict(self.prep_data['feature_data'])

    def result(self, decoder_path) -> np.ndarray:
        'translate predicted outputs to previous encoded names'
        pred = self.make_prediction()
        translator = self.load_decoder(file_path=decoder_path)
        translate = translator.inverse_transform(pred)

        return translate
