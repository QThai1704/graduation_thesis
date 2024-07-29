import torch
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt
import CGan_models_final as _model
import os
import utils
import cv2
import pickle
import networkx as nx
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.optim as optim
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.nn import BatchNorm1d

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print("Đang tải trọng số và dữ liệu hình ảnh")
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
G, result, x_1, x_2, x_3, y_1, y_2, y_3, tx_1, tx_2, ty_1, ty_2, xf_1, xf_2, yf_1, yf_2, y_n1, y_n2 = utils.load_feature_60K()
path_1 = "./fake_1/"
path_2 = "./fake_2/"
# Hãy cho đoạn code load ảnh từ thư mục

# Load ảnh từ thư mục
def load_fake_img(path):
    images = []
    for img in os.listdir(path):
        img = cv2.imread(path + img)
        images.append(img)
    return images

# X_fake1 = load_fake_img(path_1)
# X_fake2 = load_fake_img(path_2)

X_Cifar = []
X_Cifar.extend(X_train)
# X_Cifar.extend(X_fake1)
# X_Cifar.extend(X_fake2)
X_Cifar.extend(X_test)

y_Cifar = []
y_Cifar.extend(y_train)
# y_Cifar.extend(y_n1)
# y_Cifar.extend(y_n2)
y_Cifar.extend(y_test)

weight_path = './weights/transfer_resnet50_cifar100.h5'

x = []
x.extend(x_1)
x.extend(x_2)
x.extend(x_3)
# x.extend(xf_1)
# x.extend(xf_2)
x.extend(tx_1)
x.extend(tx_2)

y = []
y.extend(y_1)
y.extend(y_2)
y.extend(y_3)
# y.extend(yf_1)
# y.extend(yf_2)
y.extend(ty_1)
y.extend(ty_2)

class_names = [
    "táo",
    "cá cảnh",
    "em bé",
    "gấu",
    "hải ly",
    "giường",
    "ong",
    "bọ cánh cứng",
    "xe đạp",
    "chai",
    "bát",
    "con trai",
    "cầu",
    "xe buýt",
    "bướm",
    "lạc đà",
    "lon",
    "lâu đài",
    "sâu bướm",
    "gia súc",
    "ghế",
    "tinh tinh",
    "đồng hồ",
    "mây",
    "gián",
    "ghế sofa",
    "cua",
    "cá sấu",
    "cốc",
    "khủng long",
    "cá heo",
    "voi",
    "cá bẹt",
    "rừng",
    "cáo",
    "con gái",
    "chuột lang",
    "nhà",
    "kangaroo",
    "bàn phím",
    "đèn",
    "máy cắt cỏ",
    "báo",
    "sư tử",
    "thằn lằn",
    "tôm hùm",
    "đàn ông",
    "cây thích",
    "xe máy",
    "núi",
    "chuột",
    "nấm",
    "cây sồi",
    "cam",
    "hoa lan",
    "rái cá",
    "cây dừa",
    "lê",
    "xe bán tải",
    "cây thông",
    "đồng bằng",
    "đĩa",
    "hoa anh túc",
    "nhím",
    "thú có túi Opossum", 
    "thỏ",
    "gấu mèo",
    "cá đuối",
    "đường",
    "tên lửa",
    "hoa hồng",
    "biển",
    "hải báo",
    "cá mập",
    "chuột shrew",  
    "chồn hôi",
    "tòa nhà chọc trời",
    "ốc sên",
    "rắn",
    "nhện",
    "sóc",
    "xe điện",
    "hoa hướng dương",
    "ớt ngọt",
    "bàn",
    "xe tăng",
    "điện thoại",
    "ti vi",
    "hổ",
    "máy kéo",
    "tàu hỏa",
    "cá hồi",
    "hoa tulip",
    "rùa",
    "tủ quần áo",
    "cá voi",
    "cây liễu",
    "sói",
    "phụ nữ",
    "giun"
]
''' RESNET50 '''
# Xây dựng mô hình đầy đủ
full_model_resnet = _model._resnet50()

# Khởi tạo mô hình bằng cách chạy nó với một đầu vào giả lập
dummy_input = np.zeros((1, 32, 32, 3)) 
full_model_resnet.predict(dummy_input)

# Load model
full_model_resnet.load_weights(weight_path)

# Xây dựng mô hình chỉ để trích xuất đặc trưng
feature_model_resnet = _model._resnet50_features()

# Sao chép trọng số từ mô hình đầy đủ sang mô hình trích xuất đặc trưng
for i in range(len(feature_model_resnet.layers)):
    feature_model_resnet.layers[i].set_weights(full_model_resnet.layers[i].get_weights())

''' GCN '''
class GCNFeatureExtractor(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels):
        super(GCNFeatureExtractor, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = x.relu()
        return x

# Tạo dữ liệu đồ thị
edge_index_1 = torch.tensor([[], []], dtype=torch.long)
# Khởi tạo mô hình GCN
hidden_channels = 128
gcn_model = GCNFeatureExtractor(input_dim=256, hidden_channels=hidden_channels)
split_data = 90000
adjacency_matrix = nx.adjacency_matrix(G).tocoo()
edge_index = np.vstack([adjacency_matrix.row, adjacency_matrix.col]).astype(int)
edge_data = adjacency_matrix.data.astype(np.float32)

train_mask = np.zeros(len(x[:]), dtype=bool)
train_mask[:split_data] = True

test_mask = np.zeros(len(x[:]), dtype=bool)
test_mask[split_data:] = True

data = Data(x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            y=torch.tensor(y, dtype=torch.long),
            train_mask=torch.tensor(train_mask, dtype=torch.bool),
            test_mask=torch.tensor(test_mask, dtype=torch.bool))

gcn_model.load_state_dict(result, strict=False)

num = 30
size = 32

ob = input("Nhập đối tượng có trong tập test_img: ")
path = "./test_img/{}/".format(ob)
for i, img in enumerate(os.listdir(path)):
    print("Tra cứu ảnh thứ: ", i+1)
    count = 0
    read_img = cv2.imread(path + img)
    feature = utils._extract_feature(feature_model_resnet, read_img)
    distances = np.linalg.norm(x - feature, axis=1)
    index_min = np.argmin(distances)
    feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
    features = gcn_model(data.x, data.edge_index)
    img_test = features[index_min]
    ''' Tính khoảng cách '''
    # Tính khoảng cách Euclidean với các đặc trưng còn lại
    distances = torch.norm(features - img_test, dim=1)
    distances_dict = {}
    for m, distance in enumerate(distances):
        distances_dict[m] = distance
    # Sắp xếp từ điển theo khoảng cách từ bé tới lớn
    sorted_distances_dict = dict(sorted(distances_dict.items(), key=lambda item: item[1]))
    plt.figure(figsize=(40,20))
    plt.suptitle('Kết quả tra cứu', fontsize=42)
    for idx, n in enumerate(list(sorted_distances_dict.keys())[0:num+1]):
        # Sử dụng thư viện matplot hiển thị ảnh
        plt.subplot(9,4,1+idx)
        plt.axis('off')
        plt.imshow(X_Cifar[n])
        if idx != 0:
            if y_Cifar[n] == y_Cifar[index_min][0] and idx > 0:
                count += 1
                plt.title('1', fontsize=size)
            else:
                plt.title('0', fontsize=size)
        else:
            plt.title('Ảnh tra cứu - ' + class_names[y_Cifar[index_min][0]], fontsize=size)
        
        plt.tight_layout(pad = 4.0)
    plt.text(200, 20, 'Độ chính xác: '+ str(round(count/num,2)), ha='center', fontsize=size)
    plt.savefig('./result_60k/{}/img_{}.png'.format(ob, i+1))
print("Đã lưu kết quả tìm kiếm")
