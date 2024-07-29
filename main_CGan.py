import utils
import CGan_models as models
import train
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
import os
import matplotlib
import matplotlib.pylab as plt


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # Load data:
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    # Kích thước latent space
    latent_dim = 100
    # Khởi tạo discriminator
    d_model = models._discriminator()
    # Khởi tạo generator
    g_model = models._generator(latent_dim)
    # Khởi tạo cgan
    cgan_model = models._cgan(g_model, d_model)
    # load image data
    dataset = utils._standardize_data(X_train, y_train)
    # train model
    train._train(g_model, d_model, cgan_model, dataset, latent_dim)

