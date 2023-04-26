import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class SentimentAnalysis:

    def __init__(self, file_name, date_col):
        self.file_name = file_name
        self.date_col = date_col

    def load_data(self):
        news = pd.read_csv(self.file_name, header=0, parse_dates=[0], date_parser=self.parser)
        news.dropna(inplace=True)
        return news

    def parser(self, x):
        return datetime.datetime.strptime(x, '%Y-%m-%d')

    def clean_sent(self, sent):
        sent = sent.lower()
        special_chars2 = ['\n', '\t', '\\']
        for i in special_chars2:
            sent = sent.replace(i, ' ')
        special_chars = '~!@#$%^&*()_+{}[]:;"<>?,./\|`-='
        for i in special_chars:
            sent = sent.replace(i, '.')
        special_chars3 = "'"
        for i in special_chars3:
            sent = sent.replace(i, ' ')
        return sent.lower()

    def get_sentiment_scores(self, news_df):
        nltk.download('vader_lexicon')
        news_df['Combined'] = news_df.loc[:, news_df.columns != 'Label'].apply(lambda row: ' '.join(row[1:].values), axis=1)
        sid = SentimentIntensityAnalyzer()
        news_df.Combined = news_df.Combined.apply(self.clean_sent)
        scores = []
        for idx, val in enumerate(news_df.iterrows()):
            result = sid.polarity_scores(val[1][2])
            scores.append(result['compound'])
        news_df['scores'] = scores
        return news_df[['Date', 'scores']]
