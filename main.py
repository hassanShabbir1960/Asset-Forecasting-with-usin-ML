from data_processing import DataProcessing
from technical_indicators import TechnicalIndicators

file_name = 'upload_DJIA_table.csv'
date_col = 'Date'
cols_to_use = ['Date', 'Close', 'Volume']

# Load and process data
data = DataProcessing(file_name, date_col, cols_to_use)
dataset_ex_df = data.load_data()
data.plot_data(dataset_ex_df, 'Goldman Sachs stock price', 'USD', 'Date')

# Split into train and test datasets
train_dataset , test_dataset = data.train_test_split(dataset_ex_df)
print('Number of training days: {}. Number of test days: {}.'.format(len(train_dataset), len(test_dataset)))

# Rename 'Close' column to 'close'
dataset_ex_df.rename(columns={'Close': 'close'}, inplace=True)

# Get technical indicators
ti = TechnicalIndicators(dataset_ex_df)
dataset_TI_df = ti.get_technical_indicators()
dataset_TI_df.rename(columns={'close': 'price'}, inplace=True)

# Plot technical indicators
def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0 - last_days

    dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x_ = list(dataset.index)

    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(dataset['ma7'], label='MA 7', color='g', linestyle='--')
    plt.plot(dataset['price'], label='Closing Price', color='b')
    plt.plot(dataset['ma21'], label='MA 21', color='r', linestyle='--')
    plt.plot(dataset['upper_band'], label='Upper Band', color='c')
    plt.plot(dataset['lower_band'], label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical indicators for Goldman Sachs - last {} days.'.format(last_days))
    plt.ylabel('USD')
    plt.legend()

    # Plot second subplot
    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(dataset['MACD'], label='MACD', linestyle='-.')
    plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.plot(dataset['momentum'], label='Momentum', color='b', linestyle='-')

    plt.legend()
    plt.show()


if __name__ == '__main__':

    ##########

    from data_processing import DataProcessing
    from technical_indicators import TechnicalIndicators

    plot_technical_indicators(dataset_TI_df, 400)
    dataset_TI_df.head()


    ######### Feature Engineering

    from sentiment_analysis import SentimentAnalysis
    from wavelet_transform import WaveletTransform
    from hendometer import Hendometer
    from forex_fundamental import ForexFundamental

    # Load and preprocess sentiment analysis data
    sentiment_analysis = SentimentAnalysis('Combined_News_DJIA.csv', 'Date')
    news = sentiment_analysis.load_data()
    sentiment_scores = sentiment_analysis.get_sentiment_scores(news)

    # Load and apply wavelet transform
    wavelet_transform = WaveletTransform(dataset_TI_df, 'price')
    dataset_TI_df = wavelet_transform.apply_wavelet_transform()

    # Load and preprocess hendometer data
    hendometer = Hendometer('sumhapps.csv', 'Date')
    hendo = hendometer.load_data()

    # Load and preprocess forex fundamental data
    forex_fundamental = ForexFundamental('USDEUR.json')
    forex = forex_fundamental.load_data()
    forex = forex_fundamental.process_data(forex)

    # Merge the datasets
    dataset_TI_df = dataset_TI_df.merge(sentiment_scores, on='Date', how='outer')
    dataset_TI_df = dataset_TI_df.merge(hendo, left_on='Date', right_on='date', how='left').drop('date', axis=1)
    dataset_TI_df = dataset_TI_df.merge(forex, left_on='Date', right_on='date_fx', how='left').drop('date_fx', axis=1)
    dataset_TI_df.fillna(0.0, inplace=True)

    print(dataset_TI_df)


    #########


    from topological_analysis import TopologicalAnalysis

    # Initialize and preprocess data
    topological_analysis = TopologicalAnalysis(dataset_TI_df, 'price')
    topological_analysis.preprocess_data()

    # Apply topological analysis
    topological_analysis.apply_topological_analysis()

    # Post-process data
    dataset_TI_df = topological_analysis.post_process_data()

    print(dataset_TI_df)



    ########## TopologicalAnalysis


    import pandas as pd
    from gan import GAN
    from topological_analysis import TopologicalAnalysis
    from train_gan import TrainGan
    import tensorflow as tf

    # Constants
    HISTORICAL_DAYS_AMOUNT = 60
    SAVE_STEPS_AMOUNT = 1000
    TRAINING_AMOUNT = 10000

    # Load your dataset
    dataset_TI_df = pd.read_csv('dataforGAN.csv')  # Replace 'your_dataset.csv' with the path to your dataset

    # Create a TopologicalAnalysis instance and preprocess the dataset
    topological_analysis = TopologicalAnalysis(dataset_TI_df)
    topological_analysis.preprocess()

    # Create the GAN model
    gan_model = GAN(num_features=48, num_historical_days=HISTORICAL_DAYS_AMOUNT, generator_input_size=200)

    # Create and train the TrainGan instance
    tf.reset_default_graph()
    train_gan = TrainGan(HISTORICAL_DAYS_AMOUNT, dataset_TI_df, 128, gan_model)
    train_gan.train()



    ############## CNN


    from train_cnn import TrainCNN  # Make sure to import TrainCNN from the correct file
    import tensorflow as tf

    # Constants
    HISTORICAL_DAYS_AMOUNT = 60
    DAYS_AHEAD = 1
    PCT_CHANGE_AMOUNT = 0.01
    SAVE_STEPS_AMOUNT = 1000
    TRAINING_AMOUNT = 10000


    # Create and train the TrainCNN instance
    tf.reset_default_graph()
    train_cnn = TrainCNN(num_historical_days=HISTORICAL_DAYS_AMOUNT, days=DAYS_AHEAD, pct_change=PCT_CHANGE_AMOUNT)
    train_cnn.train()


    ################## XGBOOST

    from train_xgboost import TrainXGBBoost 

    # Create and train the TrainXGBBoost instance
    train_xgb = TrainXGBBoost(num_historical_days=20, days=10, pct_change=10)
    train_xgb.train()


    ########## GAN ESTIMATOR


    from GAN_estimator import gan_estimator
    from Predict import Predict, x1, x2

    # Call gan_estimator method
    gan_estimator.train(lambda: p.gan_predict())

    # Call Predict method
    p = Predict()
    p.gan_predict()

    # Plot results
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(150)

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(150)

    plt.plot(x, x1)
    plt.plot(x, x2)

    plt.show()
