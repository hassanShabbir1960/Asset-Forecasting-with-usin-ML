import tensorflow_gan
from TrainCNN import TrainCNN
from GAN import GAN


gan_estimator = tensorflow_gan.estimator.gan_estimator.GANEstimator(
         model_dir = '/checkpoint',
         generator_fn=GAN(num_features=13, num_historical_days=20,generator_input_size=200),
         discriminator_fn=TrainCNN(num_historical_days=20, days=5, pct_change=5),
         generator_loss_fn=tensorflow_gan.losses.wasserstein_generator_loss,
         discriminator_loss_fn=tensorflow_gan.losses.wasserstein_discriminator_loss,
         generator_optimizer=tf.compat.v1.train.AdamOptimizer(0.1, 0.5),
         discriminator_optimizer=tf.compat.v1.train.AdamOptimizer(0.1, 0.5))
