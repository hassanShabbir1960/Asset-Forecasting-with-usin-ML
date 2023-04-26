import pandas as pd
import datetime
import matplotlib.pyplot as plt

class DataProcessing:

    def __init__(self, file_name, date_col, cols_to_use):
        self.file_name = file_name
        self.date_col = date_col
        self.cols_to_use = cols_to_use

    def parser(self, x):
        return datetime.datetime.strptime(x, '%Y-%m-%d')

    def load_data(self):
        dataset_df = pd.read_csv(self.file_name, header=0, parse_dates=[0], date_parser=self.parser)
        dataset_df = dataset_df[self.cols_to_use]
        return dataset_df

    def plot_data(self, dataset_df, title, ylabel, xlabel, figsize=(14, 5), dpi=100):
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(dataset_df[self.date_col], dataset_df[self.cols_to_use[1]], label=title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()

    def train_test_split(self, dataset_df, train_ratio=0.7):
        num_training_days = int(dataset_df.shape[0] * train_ratio)
        return dataset_df[:num_training_days], dataset_df[num_training_days:]
