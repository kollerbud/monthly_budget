'Modeling and metrics here'
from abc import ABC, abstractmethod
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from text_encode import ProcessedData, LoadEncoder
from data_prep import DataCleaner, TrainCSVDataLoader


# use strategy design pattern by class composition
class Models(ABC):
    '''Train model or use pre-trained model '''
    @abstractmethod
    def __init__(self,
                 prep_data: ProcessedData,
                 model = None) -> None:
        'load the clean data from preprocessing'
        self.prep_data = prep_data
        self.model = model

    # not sure what to do yet


class TrainModel(Models):
    '''Use the random forest model'''

    x_train = None
    x_test = None
    y_train = None
    y_test = None
    model = None

    def __init__(self,
                 prep_data: ProcessedData) -> None:

        self.prep_data = prep_data

    def train_test_split(self, train_size: float = 0.75):
        'split data into train and test'
        (self.x_train, self.x_test,
         self.y_train, self.y_test) = (
            train_test_split(self.prep_data['feature_data'],
                             self.prep_data['target_data'],
                             train_size=train_size,
                             random_state=42)
            )

    def train_model(self, model = None):
        '''train model'''
        self.model = model.fit(self.x_train, self.y_train)
        return self

    def score(self):
        'output model score'
        return self.model.score(self.x_test, self.y_test)

    def save_model(self, model_name):
        'save trained model'
        joblib.dump(self.model, f'saved_models/{model_name}.joblib')


class UseModel(Models):
    '''Load and use existing models'''

    model = None

    def __init__(self,
                 prep_data: ProcessedData) -> None:
        
        self.prep_data = prep_data


    def load_model(self, model_path) -> Models:
        'load existing model'
        self.model = joblib.load(model_path)

        return self

    def load_decoder(self, file_path):
        'load decoder to translate back predictions'
        return joblib.load(filename=file_path)

    def make_prediction(self):
        'predict using model'
        return self.model.predict(self.prep_data['feature_data'])

    def result(self, decoder_path):
        'translate predicted outputs to previous encoded names'
        pred = self.make_prediction()
        translator = self.load_decoder(file_path=decoder_path)
        translate = translator.inverse_transform(pred)

        return translate


def pipeline():
    file_path = 'raw_data/activity.csv'
    i = ProcessedData(TrainCSVDataLoader(file_path=file_path),
                      cleaner=DataCleaner,
                      encoder=LoadEncoder).run()
    model = UseModel(prep_data=i).load_model(model_path='saved_models/rf.joblib')
    print(model.result(decoder_path='saved_models/category_encoder.joblib'))
    

if __name__ == '__main__':
    pipeline()





