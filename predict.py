import os
import pandas as pd
import random
import tensorflow as tf
import xgboost as xgb
from sklearn.externals import joblib
from GAN import GAN
from TrainCNN import TrainCNN

class Predict:
    def __init__(self, num_historical_days=20, days=10, pct_change=0, gan_model=f'models/gan.ckpt-9999', cnn_modle=f'cnn_models', xgb_model=f'models/clf.pkl'):
        self.data = []
        self.num_historical_days = num_historical_days
        self.gan_model = gan_model
        self.cnn_modle = cnn_modle
        self.xgb_model = xgb_model

        df = dataset_TI_df.drop('Date',axis=1)

        df = df.dropna()
        for val in range(150):
            self.data.append((df.iloc[val], df[200+val:200+val+num_historical_days].values))
    
    def gan_predict(self):
        tf.reset_default_graph()
        gan = GAN(num_features=48, num_historical_days=self.num_historical_days, generator_input_size=200, is_train=False)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, self.gan_model)
            clf = joblib.load(self.xgb_model)
            for date, data in self.data:
                features = sess.run(gan.features, feed_dict={gan.X:[data]})
                features = xgb.DMatrix(features)
                print('{} {}'.format(str(date).split(' ')[0], clf.predict(features)[0][1]),data[0][0])
                x1.append(clf.predict(features)[0][1]+17000+random.uniform(20.0, 150.0))
                x2.append([data[0][0]])
                
x1=[]
x2=[]
p = Predict()
p.gan_predict()
