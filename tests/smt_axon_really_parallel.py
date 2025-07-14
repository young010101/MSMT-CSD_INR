# 注意 4 个入参 adi，f_r，f_h, D_h 的数值范围
#
# model_param = [f_r, adi, Dh, f_csf]
# [0, 1]
# [0, 20e-6]
# [0, 1.7e-9]
# [0, 1]
import torch
from src import SMTAxonDiameterOptimized

gmr = 2.67e8

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 设置测试数据
Delta = torch.cat((torch.ones(1, 8, device=device) * 19e-3,
                   torch.ones(1, 8, device=device) * 49e-3), dim=1)
delta = torch.cat((torch.ones(1, 8, device=device) * 8e-3,
                   torch.ones(1, 8, device=device) * 8e-3), dim=1)
bvals = torch.tensor([50, 350, 800, 1500, 2400, 3450, 4750, 6000,
                      200, 950, 2300, 4250, 6750, 9850, 13500, 17800],
                     dtype=torch.float64, device=device) * 1e6

G = 1.0 / (gmr * delta) * torch.sqrt(bvals / (Delta - delta / 3))

smt_model = SMTAxonDiameterOptimized(device=device)
model_param = [0.5, 10e-6, 1.5e-9, 0.2]
smt_model.validate_parameters(model_param)
model_param_batch = torch.tensor(model_param, device=device).repeat(100, 1)
out = smt_model.forward_batch(bvals, Delta, delta, G, model_param_batch)
print(out.shape)

import matplotlib.pyplot as plt
plt.plot(out.cpu()[0, :])
plt.show()

