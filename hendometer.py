import pandas as pd
import datetime

class Hendometer:

    def __init__(self, file_name, date_col):
        self.file_name = file_name
        self.date_col = date_col

    def parser(self, x):
        return datetime.datetime.strptime(x, '%Y-%m-%d')

    def load_data(self):
        hendo = pd.read_csv(self.file_name, header=0, parse_dates=[0], date_parser=self.parser)
        return hendo
