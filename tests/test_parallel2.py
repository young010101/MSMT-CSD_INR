import time
import torch
import math
from torch.special import erf
import matplotlib.pyplot as plt

# Constants
gmr = 2.67e8


class SMTAxonDiameterOptimized:
    def __init__(self, device, max_batch_size=1000):
        self.device = device
        self.max_batch_size = max_batch_size

        # 预计算贝塞尔根
        self.bessel_root = torch.tensor([
            1.84118378134066, 5.33144277352503, 8.53631636634629, 11.7060049025921, 14.8635886339090,
            18.0155278626818, 21.1643698591888, 24.3113268572108, 27.4570505710592, 30.6019229726691,
            33.7461828986674, 36.8899874092368, 40.0334440533507, 43.1766289654488, 46.3195975611739,
            49.4623911397028, 52.6050411115567, 55.7475717922510, 58.8900022991857, 62.0323478706620
        ], device=device)

        # 预计算常量
        self.D_r = 1.7e-9
        self.D_csf = 3e-9
        self.sqrt_pi = torch.sqrt(torch.tensor(math.pi, device=device))

        # 缓存变量
        self._cached_Delta = None
        self._cached_delta = None
        self._cached_G = None
        self._cached_b = None
        self._cached_L_para = None
        self._cached_csf_signal = None

    def _update_cache(self, b, Delta, delta, G):
        """更新缓存的计算结果"""
        cache_updated = False

        if (self._cached_Delta is None or
                not torch.equal(self._cached_Delta, Delta) or
                not torch.equal(self._cached_delta, delta)):
            self._cached_Delta = Delta.clone()
            self._cached_delta = delta.clone()

            # 预计算L_para（不依赖于批次参数）
            self._cached_L_para = -(Delta - delta / 3) * (gmr * delta) ** 2 * self.D_r
            cache_updated = True

        if (self._cached_G is None or not torch.equal(self._cached_G, G)):
            self._cached_G = G.clone()
            cache_updated = True

        if (self._cached_b is None or not torch.equal(self._cached_b, b)):
            self._cached_b = b.clone()
            # 预计算CSF信号（不依赖于批次参数）
            self._cached_csf_signal = torch.exp(-b * self.D_csf).unsqueeze(0)
            cache_updated = True

        return cache_updated

    def restricted_compartment_vectorized(self, adi_batch):
        """完全向量化的受限室计算"""
        batch_size = adi_batch.shape[0]
        n_b = self._cached_Delta.shape[1]
        n_roots = len(self.bessel_root)

        # 向量化计算alphm: (batch_size, n_roots)
        adi_expanded = adi_batch.unsqueeze(1)  # (batch_size, 1)
        alphm = self.bessel_root.unsqueeze(0) / (adi_expanded / 2)  # (batch_size, n_roots)

        # 向量化计算所有alpha相关项
        alpha_2 = alphm ** 2  # (batch_size, n_roots)
        alpha_6 = alphm ** 6  # (batch_size, n_roots)

        # 扩展维度以支持广播: (batch_size, n_roots, n_b)
        alpha_2_expanded = alpha_2.unsqueeze(2)
        delta_expanded = self._cached_delta.unsqueeze(0).unsqueeze(0)
        Delta_expanded = self._cached_Delta.unsqueeze(0).unsqueeze(0)

        # 向量化计算所有因子
        factor_1 = 2 * self.D_r * alpha_2_expanded * delta_expanded
        factor_2 = 2 * torch.exp(-self.D_r * alpha_2_expanded * delta_expanded)
        factor_3 = 2 * torch.exp(-self.D_r * alpha_2_expanded * Delta_expanded)
        factor_4 = torch.exp(-self.D_r * alpha_2_expanded * (Delta_expanded - delta_expanded))
        factor_5 = torch.exp(-self.D_r * alpha_2_expanded * (Delta_expanded + delta_expanded))

        # 计算分母
        adi_expanded_2 = adi_expanded.unsqueeze(2)  # (batch_size, 1, 1)
        factor_6 = (self.D_r ** 2 * alpha_6.unsqueeze(2) *
                    ((adi_expanded_2 / 2) ** 2 * alpha_2_expanded - 1))

        # 计算E0_tmp并求和
        E0_tmp = (factor_1 - 2 + factor_2 + factor_3 - factor_4 - factor_5) / factor_6
        E0_tmp_sum = E0_tmp.sum(dim=1)  # (batch_size, n_b)

        # 计算最终信号
        L_perp = -2 * gmr ** 2 * E0_tmp_sum
        zr = self._cached_G * torch.sqrt(L_perp - self._cached_L_para)
        sig_r = (self.sqrt_pi / (2 * zr) *
                 torch.exp(self._cached_G ** 2 * L_perp) * erf(zr))

        return sig_r

    def hindered_compartment_vectorized(self, Dh_batch):
        """向量化的受阻室计算"""
        Dh_expanded = Dh_batch.unsqueeze(1)  # (batch_size, 1)
        L_perp_h = ((self._cached_delta / 3 - self._cached_Delta) *
                    (gmr * self._cached_delta) ** 2) * Dh_expanded

        zh = self._cached_G * torch.sqrt(L_perp_h - self._cached_L_para)
        sig_h = (self.sqrt_pi / (2 * zh) *
                 torch.exp(self._cached_G ** 2 * L_perp_h) * erf(zh))

        return sig_h

    def forward_batch(self, b, Delta, delta, G, model_params):
        """优化的批处理前向传播"""
        # 更新缓存
        self._update_cache(b, Delta, delta, G)

        # 解包参数
        f_r = model_params[:, 0].unsqueeze(1)  # (batch_size, 1)
        adi = model_params[:, 1] * 20e-6  # (batch_size,)
        Dh = model_params[:, 2] * 1.7e-9  # (batch_size,)
        f_csf = model_params[:, 3].unsqueeze(1)  # (batch_size, 1)

        # 计算各组分信号
        sig_r = self.restricted_compartment_vectorized(adi)
        sig_h = self.hindered_compartment_vectorized(Dh)

        # 组合信号
        forward_signal = (f_r * sig_r +
                          (1 - f_r - f_csf) * sig_h +
                          f_csf * self._cached_csf_signal)

        return forward_signal


# 使用示例和性能测试
def performance_test():
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

    # 创建优化的模型
    model_optimized = SMTAxonDiameterOptimized(device)

    # 测试不同批次大小
    batch_sizes = [1, 10, 100, 1000]

    for batch_size in batch_sizes:
        model_params = torch.tensor([0.5, 10e-6, 1.5e-9, 0.2], device=device).unsqueeze(0).repeat(batch_size, 1)

        # 预热
        for _ in range(10):
            _ = model_optimized.forward_batch(bvals, Delta, delta, G, model_params)

        # 测试优化版本
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        for _ in range(100):
            result = model_optimized.forward_batch(bvals, Delta, delta, G, model_params)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        optimized_time = time.time() - start_time

        print(f"Batch size {batch_size}:")
        print(f"  Optimized: {optimized_time:.4f}s")
        print()


if __name__ == "__main__":
    performance_test()
