# using ml flow to train model
'''Run the train pipeline or load pipeline'''
import mlflow
from data_upload import MLFlowToBucket
from encoding import MakeDataEncoder
from data_loader import DataCleaner, CSVDataLoader
from modeling import TrainModel
from sklearn.ensemble import RandomForestClassifier


# tracking info will be local, cloudsql is somewhat costly
mlflow_tracking_uri = 'sqlite:///monthly_budget.db'
mlflow.set_tracking_uri(uri=mlflow_tracking_uri)
mlflow.set_experiment(experiment_name='budget_model')


# training
def train_pipeline(raw_file_path: str):

    loader = CSVDataLoader(file_path=raw_file_path,
                           cols=['Description', 'Amount', 'City/State',
                                 'Zip Code', 'Category'])
    clean_data = DataCleaner(loader.loader_output()).cleaner_output()
    encoder = MakeDataEncoder(cleaned_data=clean_data).encoded_data()

    with mlflow.start_run():

        params = {'n_estimators': 1000,
                  'n_jobs': 6,
                  'random_state': 2,
                  }
        mlflow.log_params(params)

        use_model = RandomForestClassifier(**params)
        m = TrainModel(encoded_data=encoder, model=use_model)
        m.train_model()
        mlflow.log_param('accuracy score', m.score())
        mlflow.sklearn.log_model(use_model, artifact_path='budget')

        mlflow.sklearn.log_model(encoder['encoders'],
                                 artifact_path='vectorizer')

        mlflow.end_run()


def save_to_bucket(local_path):

    MLFlowToBucket().upload_files(
        path=local_path,
        bucket_name='monthly_budget_models')


def load_pipeline():

    return None


# train_pipeline('raw_data/activity_all.csv')
save_to_bucket(local_path='mlruns/1/dee7cad0a1e84a3cbe65d29503d74faa')
