from dataclasses import dataclass
import pandas as pd



# use strategy design pattern by class composition
@dataclass
class TrainModel:
    '''Train ML model '''
    data: DataProcess

    def train_model(self) -> None:
        pass

    def test_model(self) -> None:
        pass

    def save_model(self) -> None:
        pass

@dataclass
class UseModel:
    '''Use Pre-trained ML model'''
    data: DataProcess

    def load_model(self) -> None:
        '''load existing model'''
        pass


# training model
def load_file(file_path: str = None) -> pd.DataFrame:
    'load file from raw data'
    df = pd.read_csv('rawData/activity.csv', encoding='ISO-8859-1')

    # use relevent columns from the bank statement
    return df.loc[:, ['Description', 'Amount', 'Extended Details',
                      'City/State', 'Zip Code', 'Category']
                  ]

def text_process(col_name: list[str] = None):
    'clean up text columns'

    df_raw = load_file()

    for col in col_name:
        # remove all numbers from description, and punction symbols
        df_raw.loc[:, col] = (df_raw.loc[:, col]
                              .str.replace(r'\d+', '', regex=True)
                              .str.replace('[^\w\s]', '', regex=True)
                              .str.replace('\n', ' ', regex=True)
                              )
        # remove leading and trailing white space
        df_raw[col] = df_raw[col].str.lower()

        # remove excess white space between words
        df_raw[col] = df_raw[col].str.replace(r'\s+', ' ', regex=True)

    df_raw['Description'] = [x.replace(str(y), '')
                             for x, y in
                             zip(df_raw['Description'], df_raw['City/State'])
                             ]
    # remove leading and trailing white spaces
    df_raw['Description'] = df_raw['Description'].str.strip()

    return df_raw

def descp_encode():
    pass

# development
#load_file()
text_process(col_name=['Description', 'City/State'])