'deploy APP'
import os
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from encoding import LoadEncoderFromBucket
from modeling import LoadModelFromBucket
from data_loader import JsonDataLoader, DataCleaner
from google.cloud import aiplatform
from _auth import credentials


load_dotenv()
aiplatform.init(credentials=credentials(),
                project=os.getenv('project_id'))
model = aiplatform.Model


def run(data_or_path: str):
    '''prepare feature and target data
        runs_path should be in mlruns format,
        example: mlruns/1/12345
    '''
    loader = JsonDataLoader(
                file_path=data_or_path,
                cols=['Description', 'Amount', 'City/State',
                      'Zip Code', 'Category'])
    loader_data = loader.loader_output()
    data_cleaner = DataCleaner(data_loader=loader_data).cleaner_output()

    encoder = LoadEncoderFromBucket(clean_data=data_cleaner)
    encoder.load_vectorizer(
        bucket_name=model.list(
            'display_name="description_encoder"',
            credentials=credentials()
        )[0].uri,
        encoder_name='model.pkl'
    )

    encoder.load_target_encoder(
        bucket_name=model.list(
            'display_name="target_encoder"',
            credentials=credentials()
        )[0].uri,
        encoder_name='model.pkl'
    )
    encoded_data = encoder.encode_data()

    ml_model = LoadModelFromBucket(encoded_data=encoded_data)
    ml_model.load_bucket_model(
        bucket_name=model.list(
            'display_name="randomforest_model"',
            credentials=credentials()
        )[0].uri,
        model_name='model.pkl'
    )
    ml_model.target_encoder = encoder.target_encoder

    return ml_model.result()


# build Flask app
app = Flask(__name__)


@app.route('/predict', methods=['GET', 'POST'])
def predict_endpoint():
    'run prediction endpoint'
    data_package = request.get_json()

    pred_result = run(data_or_path=data_package)

    pred_list = pred_result.tolist()

    result = {
        'category': pred_list,
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
