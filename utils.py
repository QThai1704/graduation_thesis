import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
import matplotlib
import matplotlib.pylab as plt
import cv2
import pickle

# Hàm chuẩn hóa dữ liệu huấn luyện
def _standardize_data(X_train, y_train):
    X = X_train.astype('float32')
    X = (X - 127.5) / 127.5
    return [X, y_train]

def _generate_real_samples(dataset, n_samples):
	images, labels = dataset
	# Lựa chọn n_samples index ảnh
	ix = np.random.randint(0, images.shape[0], n_samples)
	# Lựa chọn ngẫu nhiên n_sample từ index.
	X, labels = images[ix], labels[ix]
	# print(labels.shape)
    # Khởi tạo nhãn 1 cho ảnh real
	y = np.ones((n_samples, 1))
	return [X, labels], y

# Sinh ra các véc tơ noise trong không gian latent space làm đầu vào cho generator
def _generate_latent_points(latent_dim, n_samples, n_classes=100):
	# Khởi tạo các points trong latent space
	x_input = np.random.randn(latent_dim * n_samples)
	# reshape thành batch để feed vào generator.
	z_input = x_input.reshape(n_samples, latent_dim)
	# khởi tạo labels một cách ngẫu nhiên.
	labels = np.random.randint(0, n_classes, n_samples)
	labels = Lambda(lambda x: K.expand_dims(x, axis=-1))(labels)
	return [z_input, labels]

# Sử dụng generator để sinh ra n_samples ảnh fake.
def _generate_fake_samples(generator, latent_dim, n_samples):
	# Khởi tạo các điểm ngẫu nhiên trong latent space.
	z_input, labels_input = _generate_latent_points(latent_dim, n_samples)
	# Dự đoán outputs từ generator
	images = generator.predict([z_input, labels_input])
	# Khởi tạo nhãn 0 cho ảnh fake
	y = np.zeros((n_samples, 1))
	return [images, labels_input], y

def _plot_images(images, filename, n_images):
	images = (images + 1) / 2.0 * 255
	images = images.astype(np.uint8)
	plt.figure(figsize=(12,12))
	for j in range(n_images):
			plt.subplot(12,12,1+j)
			plt.axis('off')
			plt.imshow(images[j])
	plt.savefig(filename)
	plt.show()

def _plot_losses(losses_d1, losses_d2, losses_g, filename):
	fig, axes = plt.subplots(1, 3, figsize=(12, 4))
	axes[0].plot(losses_d1)
	axes[1].plot(losses_d2)
	axes[2].plot(losses_g)
	axes[0].set_title("losses_d1")
	axes[1].set_title("losses_d2")
	axes[2].set_title("losses_g")
	plt.tight_layout()
	plt.savefig(filename)
	plt.show()

# Hàm chuẩn hóa ảnh và trích xuất đặc trưng
def _extract_feature(model, img):
    img = img * 1.0 / 255
    resized_img = cv2.resize(img, (32, 32))  # Resize về 32x32 trước khi upsample
    resized_img = resized_img.reshape((1, 32, 32, 3))
    predict_img = model.predict(resized_img)
    return predict_img[0]

def y_new(y_old):
	y = []
	for i in y_old:
		for j in range(len(i)):
			y.append(i[j])
	y = np.array(y, dtype=int)
	return y

def y_n(y):
	y_n = []
	for i in y:
		n_p = []
		n_p.append(i)
		y_n.append(n_p)
	return y_n
def load_feature_60K():
	with open('./feature/ind.cifar100_60K_135_final.graph', 'rb') as file:
		G = pickle.load(file)

	with open('./weights/gcn_60K_135_final.h5', 'rb') as file:
		result = pickle.load(file)

	# Đọc đặc trưng và nhãn từ file
	with open('./feature/ind.cifar100_1.x', 'rb') as file:
		x_1 = pickle.load(file)

	with open('./feature/ind.cifar100_2.x', 'rb') as file:
		x_2 = pickle.load(file)

	with open('./feature/ind.cifar100_3.x', 'rb') as file:
		x_3 = pickle.load(file)

	with open('./feature/ind.cifar100_1.y', 'rb') as file:
		y_1 = pickle.load(file)

	with open('./feature/ind.cifar100_2.y', 'rb') as file:
		y_2 = pickle.load(file)

	with open('./feature/ind.cifar100_3.y', 'rb') as file:
		y_3 = pickle.load(file)

	with open('./feature/ind.cifar100_1.tx', 'rb') as file:
		tx_1 = pickle.load(file)

	with open('./feature/ind.cifar100_2.tx', 'rb') as file:
		tx_2 = pickle.load(file)

	with open('./feature/ind.cifar100_1.ty', 'rb') as file:
		ty_1 = pickle.load(file)

	with open('./feature/ind.cifar100_2.ty', 'rb') as file:
		ty_2 = pickle.load(file)
	
	with open('./feature/ind.fake_1.x', 'rb') as file:
		xf_1 = pickle.load(file)

	with open('./feature/ind.fake_2.x', 'rb') as file:
		xf_2 = pickle.load(file)

	with open('./feature/ind.fake_1.y', 'rb') as file:
		yf_1 = pickle.load(file)
		yf_1 = y_new(yf_1)
		y_n1 = y_n(yf_1)

	with open('./feature/ind.fake_2.y', 'rb') as file:
		yf_2 = pickle.load(file)
		yf_2 = y_new(yf_2)
		y_n2 = y_n(yf_2)


	return [G, result, x_1, x_2, x_3, y_1, y_2, y_3, tx_1, tx_2, ty_1, ty_2, xf_1, xf_2, yf_1, yf_2, y_n1, y_n2]

