# using ml flow to train model
'''Run the train pipeline or load pipeline'''
import mlflow
from data_upload import MLFlowToBucket
from text_encode import Data_Processor, MakeEncoder
from data_loader import DataCleaner, CSVDataLoader
from model import TrainModel
from sklearn.ensemble import RandomForestClassifier


# tracking info will be local, cloudsql is somewhat costly
mlflow_tracking_uri = 'sqlite:///monthly_budget.db'
mlflow.set_tracking_uri(uri=mlflow_tracking_uri)
mlflow.set_experiment(experiment_name='budget_model')


# training
def train_pipeline(raw_file_path: str):

    loader = CSVDataLoader(file_path=raw_file_path,
                           use_cols=['Description', 'Amount', 'City/State',
                                     'Zip Code', 'Category'])
    data_processor = Data_Processor(
                      loader=loader,
                      cleaner=DataCleaner,
                      encoder=MakeEncoder,
                      encoder_path='saved_models')

    processor_data = data_processor.run()
    with mlflow.start_run():

        params = {'n_estimators': 300,
                  'n_jobs': 6,
                  'random_state': 2
                 }
        mlflow.log_params(params)

        use_model = RandomForestClassifier(**params)
        m = TrainModel(encoded_data=processor_data, model=use_model)
        m.train_model()
        mlflow.log_param('accuracy score', m.score())
        mlflow.sklearn.log_model(use_model, artifact_path='budget')

        mlflow.sklearn.log_model(processor_data['encoders'],
                        artifact_path='vectorizer')

        mlflow.end_run()

def save_to_bucket(local_path):

    MLFlowToBucket().upload_files(
        path=local_path,
        bucket_name='monthly_budget_models')

#train_pipeline('raw_data/all_year.csv')
save_to_bucket(local_path='mlruns/1/684f7cf6eef84b3b8e46dbef8e8b39a8')