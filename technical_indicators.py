import pandas as pd

class TechnicalIndicators:

    def __init__(self, dataset_df):
        self.dataset_df = dataset_df

    def get_technical_indicators(self):
        # Create 7 and 21 days Moving Average
        self.dataset_df['ma7'] = self.dataset_df['close'].rolling(window=7).mean()
        self.dataset_df['ma21'] = self.dataset_df['close'].rolling(window=21).mean()

        # Create MACD
        self.dataset_df['26ema'] = self.dataset_df['close'].ewm(com=26).mean()
        self.dataset_df['12ema'] = self.dataset_df['close'].ewm(com=12).mean()
        self.dataset_df['MACD'] = (self.dataset_df['12ema'] - self.dataset_df['26ema'])

        # Create Bollinger Bands
        self.dataset_df['20sd'] = self.dataset_df['close'].rolling(20).std()
        self.dataset_df['upper_band'] = self.dataset_df['ma21'] + (self.dataset_df['20sd'] * 2)
        self.dataset_df['lower_band'] = self.dataset_df['ma21'] - (self.dataset_df['20sd'] * 2)

        # Create Exponential moving average
        self.dataset_df['ema'] = self.dataset_df['close'].ewm(com=0.5).mean()

        # Create Momentum
        self.dataset_df['momentum'] = self.dataset_df['close'] - 1

        return self.dataset_df
