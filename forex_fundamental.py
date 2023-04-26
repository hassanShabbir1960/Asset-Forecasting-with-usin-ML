import json
import pandas as pd

class ForexFundamental:

    def __init__(self, file_name):
        self.file_name = file_name

    def load_data(self):
        USDEUR = {}
        with open(self.file_name, 'r') as fh:
            USDEUR = json.load(fh)
        forex = pd.DataFrame(columns=list(USDEUR['historical'][0].keys()))
        for idx, val in enumerate(USDEUR['historical']):
            forex.loc[idx] = list(val.values())
        return forex

    def process_data(self, forex_df):
        forex_df = forex_df[['date', 'open', 'high', 'low', 'close']]
        forex_df.columns = [str(col) + '_fx' for col in forex_df.columns]
        forex_df['date_fx'] = pd.to_datetime(forex_df['date_fx'])
        return forex_df

