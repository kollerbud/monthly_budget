'deploy APP'
import os
from flask import Flask, jsonify, request
from data_loader import JsonDataLoader, DataCleaner
from encoding import LoadEncoderFromBucket
from modeling import LoadModelFromBucket
from dotenv import load_dotenv


load_dotenv()

def run(runs_path: str, data):
    '''prepare feature and target data
        runs_path should be in mlruns format,
        example: mlruns/1/12345
    '''
    loader = JsonDataLoader(file_path=data,
                cols=['Description', 'Amount', 'City/State',
                       'Zip Code', 'Category'])
    loader_data = loader.loader_output()
    data_cleaner = DataCleaner(data_loader=loader_data).cleaner_output()

    encoder = LoadEncoderFromBucket(clean_data=data_cleaner,
                                    path=runs_path).encoded_data()
    model = LoadModelFromBucket(encoded_data=encoder,
                                model_path=runs_path)

    return model.result()


# build Flask app
app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict_endpoint():
    'run prediction endpoint'
    ml_repo_path = 'mlruns/1/684f7cf6eef84b3b8e46dbef8e8b39a8'
    data_package = request.get_json()

    pred_result = run(runs_path=ml_repo_path,
                      data=data_package)

    pred_list = pred_result.tolist()

    result = {
        'category': pred_list,
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
