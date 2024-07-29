import CGan_models_final as models
import utils
import numpy as np
import cv2
import os
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import pickle

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class_names = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm"
]

n_fake_img = 100
y_fake = []
model = models._generator(latent_dim=100, n_classes=100)
model.load_weights('./weights/cgan_generator.h5')

# Tạo các điểm ngẫu nhiên trong latent space

for i in range(0, 200):
    [z_input, labels_input] = utils._generate_latent_points(100, n_fake_img)
    # Sinh ảnh từ các điểm ngẫu nhiên
    y_fake.append(labels_input)
    X = model.predict([z_input, labels_input])
    X = X * 127.5 + 127.5
    X = X.astype(np.uint8)
    for j in range(0, n_fake_img):
        img = Image.fromarray(X[j])
        img.save(f'./fake_2/{(i*n_fake_img + j):05}.png')

with open('./feature/ind.fake_2.y', 'wb') as file:
    pickle.dump(np.array(y_fake), file)


