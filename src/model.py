'Modeling and metrics here'
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from text_encode import ProcessedData


class TrainModel:
    '''Use prepared data to train a model'''

    def __init__(self,
                 encoded_data: ProcessedData,
                 model = None) -> None:

        self._data = encoded_data.encoded()
        self.model = model
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def split(self) -> None:
        'split data into train and test'
        (self.x_train, self.x_test,
         self.y_train, self.y_test) = (
            train_test_split(self._data['feature_data'],
                             self._data['target_data'],
                             train_size=0.75,
                             random_state=42)
            )

        return self

    def score(self):
        'output model score'
        return self.model.score(self.x_test, self.y_test)

    def train_model(self):
        '''train model'''
        self.split()
        self.model.fit(self.x_train, self.y_train)
        print(f'model accuracy score is: {self.score()}')
        return self

    def save_models(self, model_loc: str = None):
        'save current model'
        pickle.dump(self.model,
                    open(f'{model_loc}/model.pkl', 'wb')
                    )
        return f'model saved to "{model_loc}"'



class UseModel:
    '''Load and use existing models'''

    model = None

    def __init__(self,
                 encoded_data: ProcessedData) -> None:
        self._data = encoded_data.transformed()

    def load_model(self, model_path):
        'load existing model'
        self.model = pickle.load(open(model_path, 'rb'))
        return self

    def load_decoder(self, file_path) -> pickle:
        'load decoder to translate back predictions'
        return pickle.load(filename=file_path)

    def make_prediction(self) -> np.ndarray:
        'predict using model'
        return self.model.predict(self.prep_data['feature_data'])

    def result(self, decoder_path) -> np.ndarray:
        'translate predicted outputs to previous encoded names'
        pred = self.make_prediction()
        translator = self.load_decoder(file_path=decoder_path)
        translate = translator.inverse_transform(pred)

        return translate