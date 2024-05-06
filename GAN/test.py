from tensorflow.keras.layers import Input, Embedding

n_classes = 10
# 1. Khởi tạo nhánh input là y_label
y_label = Input(shape=(1,))
# Embedding y_label và chiếu lên không gian véctơ 50 dimension.
y_embedding = Embedding(n_classes, 50)(y_label)
print(y_embedding)