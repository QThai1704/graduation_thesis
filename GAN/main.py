# import utils
# import train
# import models1
import tensorflow as tf
from tensorflow.keras.datasets import cifar100


if __name__ == "__main__":
    # Load data:
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    # Kích thước latent space
    latent_dim = 100
    # Khởi tạo discriminator
    d_model = models1._discriminator()
    # Khởi tạo generator
    g_model = models1._generator(latent_dim)
    # Khởi tạo cgan
    cgan_model = models1._cgan(g_model, d_model)
    # load image data
    dataset = utils._standardize_data(X_train, y_train)
    # train model
    train._train(g_model, d_model, cgan_model, dataset, latent_dim)
    # in giá trị kích thước latent space

    
