import numpy as np
from scipy.special import sph_harm
import math


class SphericalHarmonics:
    def __init__(self, l_max=8):
        """
        初始化球谐函数计算器

        Args:
            l_max: 最大阶数，默认为8
        """
        self.l_max = l_max
        self.n_coeffs = (l_max + 1) * (l_max + 2) // 2

    def real_sph_harm(self, l, m, theta, phi):
        """
        计算实值球谐函数

        Args:
            l: 阶数
            m: 相位
            theta: 极角 (0 到 π)
            phi: 方位角 (0 到 2π)

        Returns:
            实值球谐函数值
        """
        if l % 2 == 1:  # 奇数阶为0
            return 0.0

        if m < 0:
            return np.sqrt(2) * np.imag(sph_harm(-m, l, phi, theta))
        elif m == 0:
            return sph_harm(0, l, phi, theta).real
        else:
            return np.sqrt(2) * np.real(sph_harm(m, l, phi, theta))

    def get_basis_functions(self, theta, phi):
        """
        获取所有球谐基函数值

        Args:
            theta: 极角数组
            phi: 方位角数组

        Returns:
            球谐基函数矩阵 [n_directions, n_coeffs]
        """
        n_dirs = len(theta)
        Y = np.zeros((n_dirs, self.n_coeffs))

        idx = 0
        for l in range(0, self.l_max + 1, 2):  # 只考虑偶数阶
            for m in range(-l, l + 1):
                for i in range(n_dirs):
                    Y[i, idx] = self.real_sph_harm(l, m, theta[i], phi[i])
                idx += 1

        return Y

    def reconstruct_function(self, coeffs, theta, phi):
        """
        从球谐系数重建球面函数

        Args:
            coeffs: 球谐系数
            theta: 重建位置的极角
            phi: 重建位置的方位角

        Returns:
            重建的函数值
        """
        Y = self.get_basis_functions(theta, phi)
        return Y @ coeffs


# 使用示例
sh = SphericalHarmonics(l_max=8)
print(f"需要的系数个数: {sh.n_coeffs}")

# 生成测试方向
n_dirs = 100
theta = np.random.uniform(0, np.pi, n_dirs)
phi = np.random.uniform(0, 2 * np.pi, n_dirs)

# 计算球谐基函数
Y = sh.get_basis_functions(theta, phi)
print(f"基函数矩阵形状: {Y.shape}")
