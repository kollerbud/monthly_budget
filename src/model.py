'Modeling and metrics here'
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from text_encode import Data_Processor

class TrainModel:
    '''Use prepared data to train a model'''
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    def __init__(self,
                 encoded_data: Data_Processor,
                 model = None) -> None:

        self._data = encoded_data
        self.model = model


    def split(self) -> None:
        'split data into train and test'
        (self.x_train, self.x_test,
         self.y_train, self.y_test) = (
            train_test_split(self._data['feature_data'],
                             self._data['target_data'],
                             train_size=0.75,
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
        pickle.dump(self.model,
                    open(f'{model_loc}/model.pkl', 'wb')
                    )
        return f'model saved to "{model_loc}"'



class UseModel:
    '''Load and use existing models'''
    model = None
    
    def __init__(self,
                 encoded_data: ProcessedData,
                 model_path = None) -> None:

        self._data = encoded_data
        self.model_path = model_path

    def load_model(self):
        'load existing model'
        self.model = pickle.load(
            open(f'{self.model_path}/model.pkl', 'rb')
                 )

    def load_encoder(self, encoder_path) -> pickle:
        'load decoder to translate back predictions'
        return pickle.load(open(f'{encoder_path}/target_encoder.pkl', 'rb'))

    def make_prediction(self) -> np.ndarray:
        'predict using model'
        self.load_model()
        return self.model.predict(self._data['feature_data'])

    def result(self, decoder_path) -> np.ndarray:
        'translate predicted outputs to previous encoded names'
        pred = self.make_prediction()
        encoder = self.load_encoder(encoder_path=decoder_path)
        decode = encoder.inverse_transform(pred)

        return decode