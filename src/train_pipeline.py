# using ml flow to train model
'''Run the train pipeline or load pipeline'''
from encoding import MakeDataEncoder
from data_loader import DataCleaner, BigqueryDataLoader
from modeling import TrainModel
from sklearn.ensemble import RandomForestClassifier
from google.cloud import aiplatform
from _auth import credentials
import os
from dotenv import load_dotenv


load_dotenv()

aiplatform.init(credentials=credentials(),
                project=os.getenv('project_id'))


# training
def train_pipeline():

    loader = BigqueryDataLoader(months=24).loader_output()
    clean_data = DataCleaner(loader).cleaner_output()
    encoder = MakeDataEncoder(cleaned_data=clean_data)
    encoded_data = encoder.encode_data()
    encoder.save_encoders(path='./src/models')

    params = {'n_estimators': 1000,
              'n_jobs': 6,
              'random_state': 2,
              }
    use_model = RandomForestClassifier(**params)
    m = TrainModel(encoded_data=encoded_data, model=use_model)
    m.train_model()
    m.save_models(model_loc='./src/models')

    # upload encoders & models to Cloud Bucket

    model = aiplatform.Model
    model.upload_scikit_learn_model_file(
        model_file_path='./src/models/model.pkl',
        sklearn_version='1.0',
        display_name='randomforest_model',
        parent_model=model.list(
            'display_name="randomforest_model"')[0].resource_name
    )

    model.upload_scikit_learn_model_file(
        model_file_path='./src/models/desp_encoder.pkl',
        sklearn_version='1.0',
        display_name='randomforest_model',
        parent_model=model.list(
            'display_name="description_encoder"')[0].resource_name
    )

    model.upload_scikit_learn_model_file(
        model_file_path='./src/models/target_encoder.pkl',
        sklearn_version='1.0',
        display_name='randomforest_model',
        parent_model=model.list(
            'display_name="target_encoder"')[0].resource_name
    )

if __name__ == '__main__':
    train_pipeline()