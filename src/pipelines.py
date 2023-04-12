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
    
    m = TrainModel(encoded_data=i).split()
    model = RandomForestClassifier(random_state=42)
    m.train_model(model=model)
    print(model.get_params())
    #model.save_model(model_name='rf')


def use_pipeline():
    file_path = 'raw_data/activity_feb.csv'
    i = ProcessedData(TrainCSVDataLoader(file_path=file_path,
                        use_cols=['Description', 'Amount', 'City/State',
                                  'Zip Code', 'Category']),
                      cleaner=DataCleaner,
                      encoder=LoadEncoder).run()
    model = UseModel(prep_data=i)
    model.load_model(model_path='saved_models/rf.pkl')

    res = model.result(decoder_path='saved_models/category_encoder.joblib')
    print(res)

train_pipeline()