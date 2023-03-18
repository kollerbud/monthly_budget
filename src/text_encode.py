from data_prep import TrainCSVDataLoader, DataCleaning
from sklearn.feature_extraction.text import CountVectorizer
import pickle


class Encoder:
    '''encode and transform the text data from cleaned data'''
    def __init__(self, 
                 clean_data: DataCleaning = None) -> None:
        self.clean_data = clean_data.clean_output()
    
    def descp_encode(self):
        '''encode text columns'''
        # CountVectorizer
        pass

    def city_encode(self):
        pass

    def target_encode(self):
        '''encode target variable'''
        pass

def run():
    loader = TrainCSVDataLoader(file_path='rawData/activity.csv')
    cleaning = DataCleaning(loader).clean_output()
    print(cleaning)



if __name__ == '__main__':
    run()