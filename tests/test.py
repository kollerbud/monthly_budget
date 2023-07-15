import sys
from pathlib import Path
sys.path += [str(Path(__file__).parents[1])
             ]
from src.data_loader import CSVDataLoader, DataCleaner
from src.encoding import LoadEncoderFromBucket
from src.modeling import LoadModelFromBucket


def load_deployed_test():

    raw_path = 'raw_data/activity_feb.csv'
    loader_data = CSVDataLoader(file_path=raw_path,
                cols=['Description', 'Amount', 'City/State',
                      'Zip Code', 'Category']).loader_output()
    data_cleaner = DataCleaner(data_loader=loader_data).cleaner_output()

    encoder = LoadEncoderFromBucket(clean_data=data_cleaner,
                path=runs_path).encoded_data()

    model = LoadModelFromBucket(encoded_data=encoder,
                model_path=runs_path)

    return model.result()
