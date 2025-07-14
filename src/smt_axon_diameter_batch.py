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
        self._reset_cache()

    def _reset_cache(self):
        """重置所有缓存"""
        self._cached_Delta = None
        self._cached_delta = None
        self._cached_G = None
        self._cached_b = None
        self._cached_L_para = None
        self._cached_csf_signal = None
        self._cached_batch_size = None

    def _tensors_equal(self, a, b):
        """安全的张量比较"""
        if a is None or b is None:
            return False
        if a.shape != b.shape:
            return False
        return torch.allclose(a, b, rtol=1e-6, atol=1e-8)

    def _update_cache(self, b, Delta, delta, G, batch_size):
        """更新缓存的计算结果"""
        cache_updated = False

        # 检查批次大小是否改变
        if self._cached_batch_size != batch_size:
            self._cached_batch_size = batch_size
            cache_updated = True

        # 检查Delta和delta是否改变
        if (not self._tensors_equal(self._cached_Delta, Delta) or
                not self._tensors_equal(self._cached_delta, delta)):
            self._cached_Delta = Delta.clone()
            self._cached_delta = delta.clone()

            # 预计算L_para（不依赖于批次参数）
            self._cached_L_para = -(Delta - delta / 3) * (gmr * delta) ** 2 * self.D_r
            cache_updated = True

        # 检查G是否改变
        if not self._tensors_equal(self._cached_G, G):
            self._cached_G = G.clone()
            cache_updated = True

        # 检查b是否改变
        if not self._tensors_equal(self._cached_b, b):
            self._cached_b = b.clone()
            # 预计算CSF信号（不依赖于批次参数）
            self._cached_csf_signal = torch.exp(-b * self.D_csf)
            cache_updated = True

        return cache_updated

    def restricted_compartment_vectorized(self, adi_batch):
        """完全向量化的受限室计算 - 修复版本"""
        batch_size = adi_batch.shape[0]
        n_b = self._cached_Delta.shape[1]
        n_roots = len(self.bessel_root)

        # 向量化计算alphm: (batch_size, n_roots)
        adi_expanded = adi_batch.unsqueeze(1)  # (batch_size, 1)
        bessel_expanded = self.bessel_root.unsqueeze(0)  # (1, n_roots)
        alphm = bessel_expanded / (adi_expanded / 2)  # (batch_size, n_roots)

        # 向量化计算所有alpha相关项
        alpha_2 = alphm ** 2  # (batch_size, n_roots)
        alpha_6 = alphm ** 6  # (batch_size, n_roots)

        # 数值稳定性：检查 alpha_6
        # 反正越后的 Bessel root 影响越小，直接 clamp
        if torch.isnan(alpha_6).any() or torch.isinf(alpha_6).any():
            # print("警告：alpha_6 包含异常值，进行修正")
            alpha_6 = torch.clamp(alpha_6, max=1e38)  # 限制最大值

        # 扩展维度以支持广播: (batch_size, n_roots, n_b)
        alpha_2_expanded = alpha_2.unsqueeze(2)  # (batch_size, n_roots, 1)

        # 修复：正确处理delta和Delta的形状
        # self._cached_delta 和 self._cached_Delta 的形状是 (1, n_b)
        # 我们需要扩展为 (1, 1, n_b) 以便与 (batch_size, n_roots, 1) 广播
        delta_expanded = self._cached_delta.unsqueeze(1)  # (1, 1, n_b)
        Delta_expanded = self._cached_Delta.unsqueeze(1)  # (1, 1, n_b)

        # 向量化计算所有因子 - 现在形状应该是 (batch_size, n_roots, n_b)
        factor_1 = 2 * self.D_r * alpha_2_expanded * delta_expanded
        factor_2 = 2 * torch.exp(-self.D_r * alpha_2_expanded * delta_expanded)
        factor_3 = 2 * torch.exp(-self.D_r * alpha_2_expanded * Delta_expanded)
        factor_4 = torch.exp(-self.D_r * alpha_2_expanded * (Delta_expanded - delta_expanded))
        factor_5 = torch.exp(-self.D_r * alpha_2_expanded * (Delta_expanded + delta_expanded))

        # 计算分母
        adi_expanded_2 = adi_expanded.unsqueeze(2)  # (batch_size, 1, 1)
        alpha_6_expanded = alpha_6.unsqueeze(2)  # (batch_size, n_roots, 1)

        factor_6 = (self.D_r ** 2 * alpha_6_expanded *
                    ((adi_expanded_2 / 2) ** 2 * alpha_2_expanded - 1))

        # 计算E0_tmp并求和
        E0_tmp = (factor_1 - 2 + factor_2 + factor_3 - factor_4 - factor_5) / factor_6
        # E0_tmp 形状: (batch_size, n_roots, n_b)
        E0_tmp_sum = E0_tmp.sum(dim=1)  # (batch_size, n_b) - 沿着n_roots维度求和

        # 计算最终信号
        L_perp = -2 * gmr ** 2 * E0_tmp_sum  # (batch_size, n_b)
        zr = self._cached_G * torch.sqrt(L_perp - self._cached_L_para)  # (batch_size, n_b)
        sig_r = (self.sqrt_pi / (2 * zr) *
                 torch.exp(self._cached_G ** 2 * L_perp) * erf(zr))  # (batch_size, n_b)

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
        batch_size = model_params.shape[0]

        # 更新缓存
        self._update_cache(b, Delta, delta, G, batch_size)

        # 解包参数
        f_r = model_params[:, 0].unsqueeze(1)  # (batch_size, 1)
        adi = model_params[:, 1] * 20e-6  # (batch_size,)
        Dh = model_params[:, 2] * 1.7e-9  # (batch_size,)
        f_csf = model_params[:, 3].unsqueeze(1)  # (batch_size, 1)

        # 计算各组分信号
        sig_r = self.restricted_compartment_vectorized(adi)  # (batch_size, n_b)
        sig_h = self.hindered_compartment_vectorized(Dh)  # (batch_size, n_b)

        # CSF信号需要扩展到正确的批次大小
        sig_csf = self._cached_csf_signal.unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_b)

        # 组合信号 - 现在所有张量形状都应该匹配
        forward_signal = (f_r * sig_r +
                          (1 - f_r - f_csf) * sig_h +
                          f_csf * sig_csf)

        return forward_signal

    def validate_parameters(self, model_param):
        """验证模型参数是否在有效范围内"""
        f_r, adi, Dh, f_csf = model_param

        assert 0 <= f_r <= 1, f"f_r应在[0,1]范围内，当前值: {f_r}"
        assert 0 <= adi <= 20e-6, f"adi应在[0.1e-6,20e-6]范围内，当前值: {adi}"
        assert 0 <= Dh <= 1.7e-9, f"Dh应在[0.2e-9,1.7e-9]范围内，当前值: {Dh}"
        assert 0 <= f_csf <= 1, f"f_csf应在[0,1]范围内，当前值: {f_csf}"

        return True


# 进一步优化：预分配内存版本
class SMTAxonDiameterPrealloc(SMTAxonDiameterOptimized):
    def __init__(self, device, max_batch_size=1000, max_n_b=16):
        super().__init__(device, max_batch_size)
        self.max_n_b = max_n_b

        # 预分配内存缓冲区
        self._preallocated_buffers = {
            'alphm': torch.zeros(max_batch_size, len(self.bessel_root), device=device),
            'alpha_2': torch.zeros(max_batch_size, len(self.bessel_root), device=device),
            'alpha_6': torch.zeros(max_batch_size, len(self.bessel_root), device=device),
            'E0_tmp': torch.zeros(max_batch_size, len(self.bessel_root), max_n_b, device=device),
            'L_perp': torch.zeros(max_batch_size, max_n_b, device=device),
            'sig_r': torch.zeros(max_batch_size, max_n_b, device=device),
            'sig_h': torch.zeros(max_batch_size, max_n_b, device=device),
        }

    def restricted_compartment_preallocated(self, adi_batch):
        """使用预分配内存的受限室计算"""
        batch_size = adi_batch.shape[0]
        n_b = self._cached_Delta.shape[1]
        n_roots = len(self.bessel_root)

        # 检查是否超出预分配大小
        if batch_size > self.max_batch_size or n_b > self.max_n_b:
            return self.restricted_compartment_vectorized(adi_batch)

        # 使用预分配的缓冲区
        alphm = self._preallocated_buffers['alphm'][:batch_size, :n_roots]
        alpha_2 = self._preallocated_buffers['alpha_2'][:batch_size, :n_roots]
        alpha_6 = self._preallocated_buffers['alpha_6'][:batch_size, :n_roots]
        E0_tmp = self._preallocated_buffers['E0_tmp'][:batch_size, :n_roots, :n_b]
        L_perp = self._preallocated_buffers['L_perp'][:batch_size, :n_b]

        # 计算alphm
        adi_expanded = adi_batch.unsqueeze(1)
        bessel_expanded = self.bessel_root.unsqueeze(0)
        alphm.copy_(bessel_expanded / (adi_expanded / 2))

        alpha_2.copy_(alphm ** 2)
        alpha_6.copy_(alphm ** 6)

        # 其余计算保持向量化...
        alpha_2_expanded = alpha_2.unsqueeze(2)
        delta_expanded = self._cached_delta.unsqueeze(1)
        Delta_expanded = self._cached_Delta.unsqueeze(1)

        factor_1 = 2 * self.D_r * alpha_2_expanded * delta_expanded
        factor_2 = 2 * torch.exp(-self.D_r * alpha_2_expanded * delta_expanded)
        factor_3 = 2 * torch.exp(-self.D_r * alpha_2_expanded * Delta_expanded)
        factor_4 = torch.exp(-self.D_r * alpha_2_expanded * (Delta_expanded - delta_expanded))
        factor_5 = torch.exp(-self.D_r * alpha_2_expanded * (Delta_expanded + delta_expanded))

        adi_expanded_2 = adi_expanded.unsqueeze(2)
        alpha_6_expanded = alpha_6.unsqueeze(2)
        factor_6 = (self.D_r ** 2 * alpha_6_expanded *
                    ((adi_expanded_2 / 2) ** 2 * alpha_2_expanded - 1))

        E0_tmp.copy_((factor_1 - 2 + factor_2 + factor_3 - factor_4 - factor_5) / factor_6)
        E0_tmp_sum = E0_tmp.sum(dim=1)

        L_perp.copy_(-2 * gmr ** 2 * E0_tmp_sum)
        zr = self._cached_G * torch.sqrt(L_perp - self._cached_L_para)

        sig_r = self._preallocated_buffers['sig_r'][:batch_size, :n_b]
        sig_r.copy_(self.sqrt_pi / (2 * zr) *
                    torch.exp(self._cached_G ** 2 * L_perp) * erf(zr))

        return sig_r

    def forward_batch(self, b, Delta, delta, G, model_params):
        """使用预分配内存的前向传播"""
        batch_size = model_params.shape[0]
        n_b = Delta.shape[1]

        # 检查是否超出预分配大小
        if batch_size > self.max_batch_size or n_b > self.max_n_b:
            return super().forward_batch(b, Delta, delta, G, model_params)

        # 更新缓存
        self._update_cache(b, Delta, delta, G, batch_size)

        # 解包参数
        f_r = model_params[:, 0].unsqueeze(1)
        adi = model_params[:, 1] * 20e-6
        Dh = model_params[:, 2] * 1.7e-9
        f_csf = model_params[:, 3].unsqueeze(1)

        # 计算各组分信号
        sig_r = self.restricted_compartment_preallocated(adi)
        sig_h = self.hindered_compartment_vectorized(Dh)

        # CSF信号
        sig_csf = self._cached_csf_signal.unsqueeze(0).expand(batch_size, -1)

        # 组合信号
        forward_signal = (f_r * sig_r +
                          (1 - f_r - f_csf) * sig_h +
                          f_csf * sig_csf)

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
    model_prealloc = SMTAxonDiameterPrealloc(device, max_batch_size=1000)

    # 测试不同批次大小
    batch_sizes = [1, 10, 100, 1000]

    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        # adi = model_params[:, 1] * 20e-6
        # Dh = model_params[:, 2] * 1.7e-9
        model_params = torch.tensor([0.5, 0.5, 1.5 / 1.7, 0.2], device=device).unsqueeze(0).repeat(batch_size, 1)

        # 重置缓存以避免形状不匹配
        model_optimized._reset_cache()
        model_prealloc._reset_cache()

        # 预热
        try:
            for _ in range(5):
                _ = model_optimized.forward_batch(bvals, Delta, delta, G, model_params)
                _ = model_prealloc.forward_batch(bvals, Delta, delta, G, model_params)
        except Exception as e:
            print(f"预热失败: {e}")
            continue

        # 测试优化版本
        try:
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            for _ in range(100):
                result = model_optimized.forward_batch(bvals, Delta, delta, G, model_params)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            optimized_time = time.time() - start_time

            # 测试预分配版本
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            for _ in range(100):
                result = model_prealloc.forward_batch(bvals, Delta, delta, G, model_params)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            prealloc_time = time.time() - start_time

            print(f"Batch size {batch_size}:")
            print(f"  Optimized: {optimized_time:.4f}s")
            print(f"  Prealloc:  {prealloc_time:.4f}s")
            print(f"  Speedup:   {optimized_time / prealloc_time:.2f}x")
            print(f"  Result shape: {result.shape}")
            print()

        except Exception as e:
            print(f"Batch size {batch_size} 测试失败: {e}")
            print()


if __name__ == "__main__":
    performance_test()
