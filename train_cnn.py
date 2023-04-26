#training CNN

import os
import pandas as pd
import random
import tensorflow as tf
#import xgboost as xgb
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

random.seed(42)

class TrainCNN:

    def __init__(self, num_historical_days, days=10, pct_change=0):
        self.data = []
        self.labels = []
        self.test_data = []
        self.test_labels = []
        self.cnn = CNN(num_features=48, num_historical_days=num_historical_days, is_train=False)
#         files = [os.path.join('./stock_data', f) for f in os.listdir('./stock_data')]

        # Google Drive Method
        #files = [f"{googlepath}stock_data/{f}" for f in os.listdir(f"{googlepath}stock_data")]
#         print(files)
    
    
    #for file in files:
    #    print(file)
        #df = pd.read_csv(file, index_col='timestamp', parse_dates=True)
        df = dataset_TI_df.drop('Date',axis=1)
        # data for new column labels that will use the pct_change of the closing data.
        # pct_change measure change between current and prior element. Map these into a 1x2
        # array to show if the pct_change > (our desired threshold) or less than.
        labels = df.price_x.pct_change(days).map(lambda x: [int(x > pct_change/100.0), int(x <= pct_change/100.0)])
        
        # rolling normalization. (df - df.mean) / (df.max - df.min)
        #df = ((df -
        #df.rolling(num_historical_days).mean().shift(-num_historical_days))
        #/(df.rolling(num_historical_days).max().shift(-num_historical_days)
        #-df.rolling(num_historical_days).min().shift(-num_historical_days)))
        df['labels'] = labels
        df = df[:-10]
        # doing pct_change will give some rows (like first row) a NaN value. Drop that.
        df = df.dropna()
        #print(df)
        # Do the testing data split
        test_df = df[:365]
        df = df[400:]

        # get the predictors of the dataframe
        data = df.drop('labels',axis=1).values

        # the response value
        labels = df['labels'].values

        # start at num_historical_days and iterate the full length of the training
        # data at intervals of num_historical_days
        for i in range(num_historical_days, len(df), num_historical_days):
            # split the df into arrays of length num_historical_days and append
            # to data, i.e. array of df[curr - num_days : curr] -> a batch of values
            self.data.append(data[i-num_historical_days:i])

            # appending if price went up or down in curr day of "i" we are looking
            # at
            self.labels.append(labels[i-1])
        
        # do same for test data
        data = test_df.drop('labels',axis=1).values
        labels = test_df['labels'].values
        for i in range(num_historical_days, len(test_df), 1):
            self.test_data.append(data[i-num_historical_days:i])
            self.test_labels.append(labels[i-1])

    # a function to get a random_batch of data.
    def random_batch(self, batch_size=128):
        batch = []
        labels = []
        # zip concatenates each array index of both arrays together
        data = list(zip(self.data, self.labels))
        i = 0
        while True:
            i+= 1
            while True:
                # pick a random array, i.e. range of days, from data
                d = random.choice(data)
                # balance the data with equal number of positive pct_change
                # and negative pct_change
                if(d[1][0]== int(i%2)):
                    break
            batch.append(d[0])  # append the range of days we got to batch
            labels.append(d[1])  # append the label of that range of data we got
            if (len(batch) == batch_size):
                yield batch, labels
                batch = []
                labels = []

    def train(self, print_steps=100, display_steps=100, save_steps=SAVE_STEPS_AMOUNT, batch_size=128, keep_prob=0.6):
        if not os.path.exists(f'cnn_models'):
            os.makedirs(f'cnn_models')
        if not os.path.exists(f'logs'):
            os.makedirs(f'logs')
        if os.path.exists(f'logs/train'):
            for file in [os.path.join(f'logs/train/', f) for f in os.listdir(f'logs/train/')]:
                os.remove(file)
        if os.path.exists(f'logs/test'):
            for file in [os.path.join(f'logs/test/', f) for f in os.listdir(f'logs/test')]:
                os.remove(file)

        sess = tf.Session()
        loss = 0
        l2_loss = 0
        accuracy = 0
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(f'/logs/train')
        test_writer = tf.summary.FileWriter(f'/logs/test')
        sess.run(tf.global_variables_initializer())
        
        test_loss_array = []
        test_accuracy_array = []
        currentStep = "0"
        
        if os.path.exists(f'cnn_models/checkpoint'):
                with open(f'cnn_models/checkpoint', 'rb') as f:
                    model_name = next(f).split('"'.encode())[1]
                filename = "cnn_models/{}".format(model_name.decode())
                currentStep = filename.split("-")[1]
                new_saver = tf.train.import_meta_graph('{}.meta'.format(filename))
                new_saver.restore(sess, "{}".format(filename))

        for i, [X, y] in enumerate(self.random_batch(batch_size)):


            y = np.array(y)
            _, loss_curr, accuracy_curr = sess.run([self.cnn.optimizer, self.cnn.loss, self.cnn.accuracy], feed_dict=
                    {self.cnn.X:X, self.cnn.Y:y, self.cnn.keep_prob:keep_prob})
            loss += loss_curr
            accuracy += accuracy_curr
            if (i+1) % print_steps == 0:
                print('Step={} loss={}, accuracy={}'.format(i + int(currentStep), loss/print_steps, accuracy/print_steps))
                loss = 0
                l2_loss = 0
                accuracy = 0
                test_loss, test_accuracy, confusion_matrix = sess.run([self.cnn.loss, self.cnn.accuracy, self.cnn.confusion_matrix], feed_dict={self.cnn.X:self.test_data, self.cnn.Y:self.test_labels, self.cnn.keep_prob:1})
                test_loss_array.append(test_loss)
                test_accuracy_array.append(test_accuracy)
                print("Test loss = {}, Test accuracy = {}".format(test_loss, test_accuracy))
            if (i+1) % save_steps == 0:
                saver.save(sess,  f'cnn_models/cnn.ckpt', i)

            if (i+1) % display_steps == 0:
                summary = sess.run(self.cnn.summary, feed_dict=
                    {self.cnn.X:X, self.cnn.Y:y, self.cnn.keep_prob:keep_prob})
                train_writer.add_summary(summary, i)
                summary = sess.run(self.cnn.summary, feed_dict={
                    self.cnn.X:self.test_data, self.cnn.Y:self.test_labels, self.cnn.keep_prob:1})
                test_writer.add_summary(summary, i)
            
            # end training at training_amount epochs
            if (i + int(currentStep)) > TRAINING_AMOUNT:
                print("Reached {} epochs for CNN".format(i + int(currentStep)))
                sess.close()
                print(confusion_matrix)
                plot_confusion_matrix(confusion_matrix, ['Down', 'Up'], normalize=True, title="CNN Confusion Matrix")
                
                axisA = np.arange(0,len(test_loss_array),1)
                axisB = np.arange(0,len(test_accuracy_array),1)
                plt.plot(axisA, test_loss_array, label='test accuracy')
                plt.plot(axisB, test_accuracy_array, label='test loss')
                plt.legend()
                plt.title('test loss and accuracy')
                plt.show()

                break
