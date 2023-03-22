'Modeling and metrics here'
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from text_encode import ProcessedData, LoadEncoder, MakeEncoder
from data_prep import DataCleaner, TrainCSVDataLoader


class TrainModel:
    '''Use the random forest model'''

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
        return self

    def score(self):
        'output model score'
        return self.model.score(self.x_test, self.y_test)

    def save_model(self, model_name):
        'save trained model'
        joblib.dump(self.model, f'saved_models/{model_name}.joblib')


class UseModel:
    '''Load and use existing models'''

    model = None

    def __init__(self,
                 prep_data: ProcessedData) -> None:
        self.prep_data = prep_data

    def load_model(self, model_path):
        'load existing model'
        self.model = joblib.load(model_path)
        return self

    def load_decoder(self, file_path) -> joblib:
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


def train_pipeline():
    file_path = 'raw_data/all_year.csv'
    i = ProcessedData(TrainCSVDataLoader(file_path=file_path,
                        use_cols=['Description', 'Amount', 'City/State',
                                  'Zip Code', 'Category']),
                      cleaner=DataCleaner,
                      encoder=MakeEncoder).run()
    model = TrainModel(prep_data=i).split()
    model.train_model(model=RandomForestClassifier())
    model.save_model(model_name='rf')
    print(model.score())

def use_pipeline():
    file_path = 'raw_data/activity_feb.csv'
    i = ProcessedData(TrainCSVDataLoader(file_path=file_path,
                        use_cols=['Description', 'Amount', 'City/State',
                                  'Zip Code', 'Category']),
                      cleaner=DataCleaner,
                      encoder=LoadEncoder).run()
    model = UseModel(prep_data=i)
    model.load_model(model_path='saved_models/rf.joblib')

    res = model.result(decoder_path='saved_models/category_encoder.joblib')
    print(res)

if __name__ == '__main__':
    use_pipeline()