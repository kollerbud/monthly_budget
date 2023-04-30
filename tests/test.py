import sys
sys.path += ['../monthly_budget', '../monthly_budget/src']
from src.data_loader import CSVDataLoader, DataCleaner
from src.encoding import LoadEncoderFromBucket
from modeling import LoadModelFromBucket


def load_deployed_test():

    raw_path = 'raw_data/activity_feb.csv'
    loader_data = CSVDataLoader(file_path=raw_path,
                cols=['Description', 'Amount', 'City/State',
                      'Zip Code', 'Category']).loader_output()
    data_cleaner = DataCleaner(data_loader=loader_data).cleaner_output()

    # pick which runs to use
    runs_path = 'mlruns/1/684f7cf6eef84b3b8e46dbef8e8b39a8'

    encoder = LoadEncoderFromBucket(clean_data=data_cleaner,
                path=runs_path).encoded_data()

    model = LoadModelFromBucket(encoded_data=encoder,
                model_path=runs_path)

    return model.result()

print(load_deployed_test())
