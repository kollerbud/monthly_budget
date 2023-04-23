from flask import Flask, jsonify, request
from data_loader import JsonDataLoader, DataCleaner
from text_encode import Data_Processor, LoadEncoder
from model import UseModel

# load raw data
# prepare and encode feature data
def prepare_feature(encoder_path):
    'prepare feature and target data'

    loader = JsonDataLoader(file_path=request.get_json(),
                use_cols=['Description', 'Amount', 'City/State',
                                  'Zip Code', 'Category'])

    process_data = Data_Processor(loader=loader,
                        cleaner=DataCleaner,
                        encoder=LoadEncoder,
                        encoder_path=encoder_path)

    return process_data.run()

# make prediction
def predict(feature_data, model_path, decoder_path):

    model = UseModel(feature_data, model_path=model_path)
    model.make_prediction()
    result = model.result(decoder_path=decoder_path)

    return result

app = Flask('budget category prediction')


@app.route('/predict', methods=['GET'])
def predict_endpoint():

    features = prepare_feature(encoder_path='saved_models')

    pred = predict(feature_data=features,
                   model_path='saved_models',
                   decoder_path='saved_models')

    pred_list = pred.tolist()

    result = {
        'category': pred_list
    }


    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)