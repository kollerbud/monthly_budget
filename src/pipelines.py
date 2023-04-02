'''Run the train pipeline or load pipeline'''
from text_encode import ProcessedData, LoadEncoder, MakeEncoder
from data_prep import DataCleaner, TrainCSVDataLoader
from model import TrainModel, UseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_pipeline():
    file_path = 'raw_data/all_year.csv'
    i = ProcessedData(TrainCSVDataLoader(file_path=file_path,
                        use_cols=['Description', 'Amount', 'City/State',
                                  'Zip Code', 'Category']),
                      cleaner=DataCleaner,
                      encoder=MakeEncoder).run()
    model = TrainModel(prep_data=i).split()
    model.train_model(model=LogisticRegression())
    model.save_model(model_name='lr')
    print(model.score())

def use_pipeline():
    file_path = 'raw_data/activity_feb.csv'
    i = ProcessedData(TrainCSVDataLoader(file_path=file_path,
                        use_cols=['Description', 'Amount', 'City/State',
                                  'Zip Code', 'Category']),
                      cleaner=DataCleaner,
                      encoder=LoadEncoder).run()
    model = UseModel(prep_data=i)
    model.load_model(model_path='saved_models/lr.joblib')

    res = model.result(decoder_path='saved_models/category_encoder.joblib')
    print(res)

use_pipeline()