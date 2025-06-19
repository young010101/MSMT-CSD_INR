import numpy as np


class SphericalHarmonics:
    def __init__(self, l_max=8):
        self.l_max = l_max
        self.n_coeffs = (l_max + 1) * (l_max + 2) // 2

    def real_sph_harm(self, l, m, theta, phi) -> np.ndarray:
        """
        计算实值球谐函数
        """
        pass

    def get_basis_functions(self, theta, phi):
        """
        获取所有球谐基函数值
        """
        pass

    def reconstruct(self, coeffs, theta, phi):
        """根据系数和方向，重建函数值.

        系数加上球谐基函数矩阵相乘
        """
        pass

sh = SphericalHarmonics(l_max=8)
print(f"需要的系数个数: {sh.n_coeffs}")

# 生成测试方向
n_dirs = 100
theta = np.random.uniform(0, np.pi, n_dirs)
phi = np.random.uniform(0, 2 * np.pi, n_dirs)

# 可视化这些方向
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
ax.set_title("测试方向")
plt.show()

