import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
import cv2
import networkx as nx

# Tải bộ dữ liệu CIFAR-100
cifar100 = tf.keras.datasets.cifar100
(X_train, Y_train), (X_test, _) = cifar100.load_data()


# Mô hình trích xuất đặc trưng ResNet50
def model_resnet50():
    resnet_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = resnet_model.output
    x_newfc = x = GlobalAveragePooling2D()(x)
    feature_extractor = tf.keras.Model(inputs=resnet_model.input, outputs=x_newfc)
    return feature_extractor

# Hàm chuẩn  hóa ảnh và trích xuất đặc trưng
def extract_feature(X, index):
    feature_extractor = model_resnet50()
    img = X[index]
    img = X[index] * 1.0 /255
    resized_img = cv2.resize(img, (224, 224))
    resized_img = resized_img.reshape((1, 224, 224, 3))
    predict_img = feature_extractor.predict(resized_img)
    feature = predict_img[0].tolist()
    label = int(Y_train[index][0])
    return feature, label

# Xây dựng đồ thị
# Khai báo các biến cần thiết
dict = {}

# Hàm tính khoảng cách giữa hai đặc trưng
def distance(feature1, feature2):
    return np.linalg.norm(np.array(feature1) - np.array(feature2))

# Hàm sắp xếp
def sort_lists(list_dict, link):
    combined = list(zip(list_dict, link))
    combined.sort()
    sorted_list, sorted_link = zip(*combined)
    return list(sorted_list), list(sorted_link)

# Hàm tạo các node (train, test) cho đồ thị
def graph_node():
    # for i in range(len(X_train)):
    for i in range(100):
        feature_train, labels_train = extract_feature(X_train, i)
        G.add_node(int(i), feature=feature_train)
        dict[i] = labels_train
    # a = len(G.nodes())
    # for j in range(1):
    #     feature_test, _ = extract_feature(X_test, j)
    #     G.add_node(int(a), feature=feature_test)
    #     a = a + 1
G = nx.Graph()
graph_node()
features = []
labels = []


# Hàm thêm cạnh cho các node train
def add_edge_train(array):
    for i in range(len(array)):
        for j in range(i, len(array)):
            G.add_edge(array[i], array[j])

# Hàm xây dựng cạnh cho các node trong train
def graph_edge_train():
    for i in range(100):
        arr = []
        for key, value in dict.items():
            if i == value:
                arr.append(key)
        add_edge_train(arr)

# Hàm xây dựng cạnh cho các node trong cây trong test
# def graph_edge_test():
#     a = 100
#     list_dict = []
#     link = []
#     # for i in range(len(X_test)):
#     for i in range(1):
#         G.add_edge(a, a)
#         for j in range(100):
#             X_test_feature, _ = extract_feature(X_test, i)
#             dist = distance(X_test_feature, G.nodes[j]['feature'])
#             list_dict.append(dist)
#             link.append([a, j])
#         _, sorted_link = sort_lists(list_dict, link)
#         for k in range(10):
#             G.add_edge(sorted_link[k][0], sorted_link[k][1])
#         a = a + 1
#     return G

graph_edge_train()
# G = graph_edge_test()

# Save the data dictionary using libraries like pickle or joblib
import pickle
with open('graph_data.pkl', 'wb') as f:
    pickle.dump(data, f)

# Lưu cây vào file
nx.write_gml(G, "graph.gml")

# Draw the graph
node_of_interest_1 = 100
node_of_interest_2 = 37
def draw_graph(node_of_interest_1, node_of_interest_2, G):
    # Kiểm tra xem các node có tồn tại trong đồ thị không
    for node_of_interest in [node_of_interest_1, node_of_interest_2]:
        if node_of_interest not in G:
            print(f"Node {node_of_interest} không tồn tại trong đồ thị.")
        else:
            # Lấy các node kề của node_of_interest và thêm vào danh sách node cần vẽ
            neighbors = list(G.neighbors(node_of_interest))
            subgraph_nodes = [node_of_interest] + neighbors

            # Lấy tất cả các node kề của các node kề của node_of_interest và thêm vào danh sách node cần vẽ
            for neighbor in neighbors:
                subgraph_nodes.extend(list(G.neighbors(neighbor)))

            # Loại bỏ các node trùng lặp và tạo một đồ thị con mới
            subgraph_nodes = list(set(subgraph_nodes))
            subgraph = G.subgraph(subgraph_nodes)

            # Vẽ đồ thị con
            pos = nx.spring_layout(subgraph)  # Tạo layout cho đồ thị con
            plt.figure(figsize=(8, 6))
            nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=12)

            # Làm nổi bật node_of_interest
            nx.draw_networkx_nodes(subgraph, pos, nodelist=[node_of_interest], node_color='red', node_size=800)

            plt.title(f"Subgraph with Node {node_of_interest} and its Neighbors")
            plt.show()
    
draw_graph(node_of_interest_1, node_of_interest_2, G)




