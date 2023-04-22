'''Run the train pipeline or load pipeline'''
from text_encode import Data_Processor, LoadEncoder, MakeEncoder
from data_loader import DataCleaner, CSVDataLoader
from model import TrainModel, UseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_pipeline():
    file_path = 'raw_data/all_year.csv'
    loader = CSVDataLoader(file_path=file_path,
                            use_cols=['Description', 'Amount', 'City/State',
                              'Zip Code', 'Category'])
    data_processor = Data_Processor(loader=loader,
                      cleaner=DataCleaner,
                      encoder=MakeEncoder,
                      encoder_path='saved_models')
    processor_data = data_processor.run()
    data_processor.save_encoder()
    
    use_model = RandomForestClassifier()
    m = TrainModel(encoded_data=processor_data, model=use_model)
    m.train_model()
    m.save_models(model_loc='saved_models')
    
    print(use_model.get_params())


def use_pipeline():
    file_path = 'raw_data/activity_feb.csv'
    loader = TrainCSVDataLoader(file_path=file_path,
                                use_cols=['Description', 'Amount', 'City/State',
                                  'Zip Code', 'Category'])    
    i = ProcessedData(loader=loader,
                      cleaner=DataCleaner,
                      encoder=LoadEncoder,
                      encoder_path='saved_models').run()
    model_path = 'saved_models'
    use_model = UseModel(encoded_data=i, model_path=model_path)
    use_model.make_prediction()
    res = use_model.result(decoder_path='saved_models')
    print(res)

train_pipeline()