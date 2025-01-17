import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, LeakyReLU, ReLU, Embedding, Concatenate, Reshape, Flatten, Dropout, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, UpSampling2D, Dropout
from tensorflow.keras.models import Sequential

def _discriminator(input_shape=(32, 32, 3), n_classes = 100):
	# 1. Khởi tạo nhánh input là y_label
	y_label = Input(shape=(1,))
	# Embedding y_label và chiếu lên không gian véc tơ 65 kích thước.
	y_embedding = Embedding(n_classes, 50)(y_label)
	# Gia tăng kích thước y_embedding thông qua linear projection
	n_shape = input_shape[0] * input_shape[1]
	li = Dense(n_shape)(y_embedding)
	li = Reshape((input_shape[0], input_shape[1], 1))(li)

	# 2. Khởi tạo nhánh input là image
	inpt_image = Input(shape=input_shape)

	# 3. Concate y_label và image
	concat = Concatenate()([inpt_image, li])

	# 4. Feature extractor thông qua CNN blocks:
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(concat) #16x16x128
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe) #8x8x128
	fe = LeakyReLU(alpha=0.2)(fe)

	# Flatten output
	fe = Flatten()(fe)
	fe = Dropout(0.4)(fe)
	out_layer = Dense(1, activation='sigmoid')(fe)

	# Khởi tạo model
	model = Model([inpt_image, y_label], out_layer)
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

def _generator(latent_dim, n_classes=100):
	# 1. Khởi tạo nhánh đầu vào là y_label
	y_label = Input(shape=(1,))
	# embedding véc tơ phân loại đầu vào
	li = Embedding(n_classes, 50)(y_label)
	n_shape = 8 * 8
	li = Dense(n_shape)(li)
	# reshape lại đầu vào về kích thước 2x2x1 như một channel bổ sung.
	li = Reshape((8, 8, 1))(li)

	# 2. Khởi tạo nhánh đầu vào là véc tơ noise x
	in_lat = Input(shape=(latent_dim,))
	n_shape = 8*8*128
	gen = Dense(n_shape)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((8, 8, 128))(gen)

	# 3. Merge nhánh 1 và nhánh 2
	merge = Concatenate()([gen, li])
	# 4. Sử dụng Conv2DTranspose để giải chập về kích thước ban đầu.
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge) #16x16x128
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 32x32
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen) #32x32x128
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Conv2D(3, (8,8), activation='tanh', padding='same')(gen) #32x32x3

	# model
	model = Model([in_lat, y_label], out_layer)
	return model

def _cgan(g_model, d_model):
	# Do cgan được sử dụng để huấn luyện generator nên discriminator sẽ được đóng băng
	d_model.trainable = False
	# Lấy đầu vào của generator model bao gồm véc tơ noise và nhãn
	gen_noise, gen_label = g_model.input
	# Lấy ảnh sinh ra từ generator model
	gen_output = g_model.output
	# Truyền output và nhãn của mô hình generator vào mô hình discriminator
	gan_output = d_model([gen_output, gen_label])
	# Khởi tạo mô hình CGAN
	model = Model([gen_noise, gen_label], gan_output)
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

def _resnet50():
    resnet_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    for layer in resnet_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

    model = Sequential()
    model.add(UpSampling2D(size=(7, 7), interpolation='bilinear', input_shape=(32, 32, 3)))
    model.add(resnet_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dense(100, activation='softmax'))
    
    return model

# Xây dựng mô hình chỉ để trích xuất đặc trưng
def _resnet50_features():
    resnet_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    for layer in resnet_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

    model = Sequential()
    model.add(UpSampling2D(size=(7, 7), interpolation='bilinear', input_shape=(32, 32, 3)))
    model.add(resnet_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    
    return model