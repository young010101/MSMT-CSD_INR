from src import smt_axon_diameter, gmr
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import torch

start_time = time.time()

model_param = [0.5, 10e-6, 1.5e-9, 0.2]

Delta = np.concatenate((np.ones([1, 8]) * 19e-3, np.ones([1, 8]) * 49e-3), axis=1)
delta = np.concatenate((np.ones([1, 8]) * 8e-3, np.ones([1, 8]) * 8e-3), axis=1)
bvals = np.array([50, 350, 800, 1500, 2400, 3450, 4750, 6000, 200, 950, 2300, 4250, 6750, 9850, 13500, 17800]) * 1e6
G = 1. / (gmr * delta) * np.sqrt(bvals / (Delta - delta / 3))
q = (1 / 2 / math.pi) * gmr * G * delta

for i in range(10000):
    smt_axon_diameter(bvals, Delta, delta, G, model_param)
# sig = smt_axon_diameter(bvals, Delta, delta, G, model_param)
end_time = time.time()

print("Time: ", end_time - start_time)

# plt.plot(np.arange(1, 9), sig[0, 0:8], color='blue', marker='o')
# plt.plot(np.arange(1, 9), sig[0, 8:], color='red', marker='s')
#
# plt.grid(True)
# plt.show()

coeffs = torch.randn(100, 45)
conv_vec = torch.randn(45)
y_mat = torch.randn(724, 45)

start_time = time.time()
# batch_size x 45, 45, 724 x 45
torch.einsum("bk, k, dk -> bd", coeffs, conv_vec, y_mat)
end_time = time.time()
print("Time: ", end_time - start_time)

# %% 测试不同batch size 的运行时间增长曲线
import torch
import time

for batch_size in [100, 1000, 10000, 100000, 1000000, 10000000]:
    coeffs = torch.randn(batch_size, 45)
    conv_vec = torch.randn(45)
    y_mat = torch.randn(724, 45)
    start_time = time.time()
    torch.einsum("bk, k, dk -> bd", coeffs, conv_vec, y_mat)
    end_time = time.time()
    print("Time: ", end_time - start_time)

    # Time: 0.00028204917907714844
    # Time: 0.0005629062652587891
    # Time: 0.004350900650024414
    # Time: 0.029003143310546875
    # Time: 0.34634900093078613
    # Time: 15.025325059890747

# %%
import torch
import time
from src import smt_axon_diameter_tensor

sig = smt_axon_diameter_tensor(bvals, Delta, delta, G, model_param)