from tensorflow.keras.layers import Input, Embedding
import models

latent_dim = 100
d_model = models._discriminator()
# Lấy đầu vào của generator model bao gồm véc tơ noise và nhãn
gen_noise1, gen_label1 = d_model.input
# Lấy ảnh sinh ra từ generator model
gen_output1 = d_model.output
print(gen_noise1)
print(gen_label1)
print(gen_output1)