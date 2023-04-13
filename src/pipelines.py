'''Run the train pipeline or load pipeline'''
from text_encode import ProcessedData, LoadEncoder, MakeEncoder
from data_prep import DataCleaner, TrainCSVDataLoader
from model import TrainModel, UseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_pipeline():
    file_path = 'raw_data/all_year.csv'
    loader = TrainCSVDataLoader(file_path=file_path,
                                use_cols=['Description', 'Amount', 'City/State',
                                  'Zip Code', 'Category'])
    i = ProcessedData(loader=loader,
                      cleaner=DataCleaner, encoder=LoadEncoder, file_loc='saved_models/')
    
    m = TrainModel(encoded_data=i, model=RandomForestClassifier())
    print(m.get_params())
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