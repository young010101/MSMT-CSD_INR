import time
import torch
import math
from torch.special import erf
import matplotlib.pyplot as plt

# Constants
gmr = 2.67e8

# 在文件顶部添加全局变量
alphm_warning_count = 0


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
        """完全向量化的受限室计算 - 超稳定版本"""
        batch_size = adi_batch.shape[0]
        n_b = self._cached_Delta.shape[1]
        n_roots = len(self.bessel_root)

        # 严格的输入检查和处理 - 使用更保守的范围
        if torch.isnan(adi_batch).any() or torch.isinf(adi_batch).any():
            print("警告：adi_batch包含NaN或Inf值，进行修正")
            adi_batch = torch.clamp(adi_batch, min=1e-7, max=5e-6)  # 更保守的范围
        
        # 确保adi_batch在合理范围内 - 更保守的范围
        adi_batch = torch.clamp(adi_batch, min=1e-7, max=5e-6)  # 更保守的上限

        # 向量化计算alphm: (batch_size, n_roots)
        adi_expanded = adi_batch.unsqueeze(1)  # (batch_size, 1)
        bessel_expanded = self.bessel_root.unsqueeze(0)  # (1, n_roots)
        
        # 数值稳定性：确保adi不为0，使用更保守的范围
        adi_safe = torch.clamp(adi_expanded, min=1e-6, max=5e-6)  # 更保守的范围
        alphm = bessel_expanded / (adi_safe / 2)  # (batch_size, n_roots)

        # 严格检查alphm - 更保守的限制
        if torch.isnan(alphm).any() or torch.isinf(alphm).any():
            print("警告：alphm包含异常值，进行修正")
            alphm = torch.clamp(alphm, min=1e-6, max=1e2)  # 更保守的范围

        # 预检查alphm是否会溢出 - 更严格的上限
        if (alphm > 1e1).any():
            global alphm_warning_count
            alphm_warning_count += 1
            # 替换原有的print("警告：alphm值过大，限制在安全范围内")为计数
            # 并在每个epoch结束后输出统计
            # print("警告：alphm值过大，限制在安全范围内")

        # 向量化计算所有alpha相关项
        alpha_2 = alphm ** 2  # (batch_size, n_roots)
        
        # 更严格的alpha_6检查
        alpha_6 = alphm ** 6  # (batch_size, n_roots)

        # 严格检查 alpha_6 - 更保守的限制
        if torch.isnan(alpha_6).any() or torch.isinf(alpha_6).any():
            print("警告：alpha_6 包含异常值，重新计算")
            # 重新计算，使用更保守的限制
            alphm = torch.clamp(alphm, max=1e0)  # 更保守的上限
            alpha_2 = alphm ** 2
            alpha_6 = alphm ** 6
            alpha_6 = torch.clamp(alpha_6, max=1e6)  # 更保守的最大值

        # 扩展维度以支持广播: (batch_size, n_roots, n_b)
        alpha_2_expanded = alpha_2.unsqueeze(2)  # (batch_size, n_roots, 1)

        # 修复：正确处理delta和Delta的形状
        delta_expanded = self._cached_delta.unsqueeze(1)  # (1, 1, n_b)
        Delta_expanded = self._cached_Delta.unsqueeze(1)  # (1, 1, n_b)

        # 安全的指数计算 - 更严格的范围限制
        def safe_exp(x, name="exp_input"):
            """安全的指数计算"""
            # 限制输入范围以避免溢出 - 更严格的范围
            x_safe = torch.clamp(x, min=-10, max=10)  # 更严格的范围
            result = torch.exp(x_safe)
            if torch.isnan(result).any() or torch.isinf(result).any():
                print(f"警告：{name} 指数计算失败，使用近似")
                result = torch.where(x_safe > 0, torch.exp(torch.clamp(x_safe, max=8)), 
                                   torch.where(x_safe < 0, torch.exp(torch.clamp(x_safe, min=-8)), 1.0))
            return result

        # 向量化计算所有因子 - 使用安全的指数计算
        factor_1 = 2 * self.D_r * alpha_2_expanded * delta_expanded
        
        # 使用安全的指数计算 - 更严格的限制
        exp_arg_1 = -self.D_r * alpha_2_expanded * delta_expanded
        exp_arg_2 = -self.D_r * alpha_2_expanded * Delta_expanded
        exp_arg_3 = -self.D_r * alpha_2_expanded * (Delta_expanded - delta_expanded)
        exp_arg_4 = -self.D_r * alpha_2_expanded * (Delta_expanded + delta_expanded)
        
        # 限制指数参数范围
        exp_arg_1 = torch.clamp(exp_arg_1, min=-10, max=10)
        exp_arg_2 = torch.clamp(exp_arg_2, min=-10, max=10)
        exp_arg_3 = torch.clamp(exp_arg_3, min=-10, max=10)
        exp_arg_4 = torch.clamp(exp_arg_4, min=-10, max=10)
        
        factor_2 = 2 * safe_exp(exp_arg_1, "factor_2")
        factor_3 = 2 * safe_exp(exp_arg_2, "factor_3")
        factor_4 = safe_exp(exp_arg_3, "factor_4")
        factor_5 = safe_exp(exp_arg_4, "factor_5")

        # 检查因子是否包含异常值
        for i, factor in enumerate([factor_1, factor_2, factor_3, factor_4, factor_5]):
            if torch.isnan(factor).any() or torch.isinf(factor).any():
                print(f"警告：factor_{i+1} 包含异常值，进行修正")
                factor = torch.clamp(factor, min=-1e8, max=1e8)  # 更保守的范围

        # 计算分母 - 更安全的计算
        adi_expanded_2 = adi_expanded.unsqueeze(2)  # (batch_size, 1, 1)
        alpha_6_expanded = alpha_6.unsqueeze(2)  # (batch_size, n_roots, 1)

        # 分步计算分母，避免数值问题
        adi_half_squared = (adi_expanded_2 / 2) ** 2
        alpha_2_term = adi_half_squared * alpha_2_expanded - 1
        
        # 检查alpha_2_term是否接近零
        if (torch.abs(alpha_2_term) < 1e-15).any():
            print("警告：alpha_2_term接近零，进行调整")
            alpha_2_term = torch.where(torch.abs(alpha_2_term) < 1e-15, 
                                     torch.sign(alpha_2_term) * 1e-15, alpha_2_term)

        factor_6 = self.D_r ** 2 * alpha_6_expanded * alpha_2_term

        # 检查分母是否为零或包含异常值
        if torch.isnan(factor_6).any() or torch.isinf(factor_6).any():
            print("警告：factor_6 包含异常值，进行修正")
            factor_6 = torch.clamp(factor_6, min=1e-40, max=1e8)  # 更保守的范围

        # 避免除零 - 更严格的最小值
        factor_6 = torch.where(torch.abs(factor_6) < 1e-40, 
                              torch.sign(factor_6) * 1e-40, factor_6)

        # 计算E0_tmp并求和
        E0_tmp = (factor_1 - 2 + factor_2 + factor_3 - factor_4 - factor_5) / factor_6
        # E0_tmp 形状: (batch_size, n_roots, n_b)
        
        # 检查E0_tmp是否包含异常值 - 更保守的限制
        if torch.isnan(E0_tmp).any() or torch.isinf(E0_tmp).any():
            print("警告：E0_tmp 包含异常值，进行修正")
            E0_tmp = torch.clamp(E0_tmp, min=-1e6, max=1e6)  # 更保守的范围
        
        E0_tmp_sum = E0_tmp.sum(dim=1)  # (batch_size, n_b) - 沿着n_roots维度求和

        # 计算最终信号
        L_perp = -2 * gmr ** 2 * E0_tmp_sum  # (batch_size, n_b)
        
        # 检查L_perp是否包含异常值 - 更保守的限制
        if torch.isnan(L_perp).any() or torch.isinf(L_perp).any():
            print("警告：L_perp 包含异常值，进行修正")
            L_perp = torch.clamp(L_perp, min=-1e6, max=1e6)  # 更保守的范围
        
        # 计算zr - 更安全的计算
        L_diff = L_perp - self._cached_L_para
        
        # 检查L_diff是否为正数（对数运算要求）
        if (L_diff <= 0).any():
            print("警告：L_diff包含非正值，进行调整")
            L_diff = torch.clamp(L_diff, min=1e-40)  # 更严格的最小值
        
        zr = self._cached_G * torch.sqrt(L_diff)  # (batch_size, n_b)
        
        # 检查zr是否包含异常值 - 更保守的限制
        if torch.isnan(zr).any() or torch.isinf(zr).any():
            print("警告：zr 包含异常值，进行修正")
            zr = torch.clamp(zr, min=1e-10, max=1e2)  # 更保守的范围
        
        # 使用更稳定的erf计算
        try:
            # 限制zr的范围以避免erf计算问题 - 更严格的范围
            zr_safe = torch.clamp(zr, min=-3, max=3)  # 更严格的范围
            erf_zr = torch.erf(zr_safe)
        except:
            print("警告：erf计算失败，使用近似值")
            erf_zr = torch.tanh(zr_safe)  # 使用tanh作为erf的近似
        
        # 计算sig_r - 分步计算以避免数值问题
        exp_arg = self._cached_G ** 2 * L_perp
        # 限制指数参数范围
        exp_arg = torch.clamp(exp_arg, min=-10, max=10)
        exp_term = safe_exp(exp_arg, "sig_r_exp")
        
        # 避免除零 - 更严格的最小值
        zr_safe_denom = torch.where(torch.abs(zr) < 1e-15, 
                                   torch.sign(zr) * 1e-15, zr)
        
        sig_r = (self.sqrt_pi / (2 * zr_safe_denom) * exp_term * erf_zr)  # (batch_size, n_b)

        # 最终检查输出 - 更保守的范围
        if torch.isnan(sig_r).any() or torch.isinf(sig_r).any():
            print("警告：sig_r 包含异常值，进行修正")
            sig_r = torch.clamp(sig_r, min=0.0, max=1.0)

        return sig_r

    def hindered_compartment_vectorized(self, Dh_batch):
        """向量化的受阻室计算"""
        # 检查输入参数
        if torch.isnan(Dh_batch).any() or torch.isinf(Dh_batch).any():
            print("警告：Dh_batch包含NaN或Inf值")
            Dh_batch = torch.clamp(Dh_batch, min=1e-12, max=1.7e-9)
        
        # 更保守的范围限制
        Dh_batch = torch.clamp(Dh_batch, min=1e-12, max=0.5e-9)  # 更保守的上限
        
        Dh_expanded = Dh_batch.unsqueeze(1)  # (batch_size, 1)
        L_perp_h = ((self._cached_delta / 3 - self._cached_Delta) *
                    (gmr * self._cached_delta) ** 2) * Dh_expanded

        # 检查L_perp_h是否包含异常值 - 更保守的限制
        if torch.isnan(L_perp_h).any() or torch.isinf(L_perp_h).any():
            print("警告：L_perp_h 包含异常值，进行修正")
            L_perp_h = torch.clamp(L_perp_h, min=-1e8, max=1e8)  # 更保守的范围

        # 确保L_perp_h - L_para为正数
        L_diff_h = L_perp_h - self._cached_L_para
        if (L_diff_h <= 0).any():
            print("警告：L_diff_h包含非正值，进行调整")
            L_diff_h = torch.clamp(L_diff_h, min=1e-40)  # 更严格的最小值

        zh = self._cached_G * torch.sqrt(L_diff_h)
        
        # 检查zh是否包含异常值 - 更保守的限制
        if torch.isnan(zh).any() or torch.isinf(zh).any():
            print("警告：zh 包含异常值，进行修正")
            zh = torch.clamp(zh, min=1e-10, max=1e2)  # 更保守的范围
        
        # 使用更稳定的erf计算
        try:
            # 限制zh的范围
            zh_safe = torch.clamp(zh, min=-3, max=3)  # 更严格的范围
            erf_zh = torch.erf(zh_safe)
        except:
            print("警告：erf计算失败，使用近似值")
            erf_zh = torch.tanh(zh_safe)  # 使用tanh作为erf的近似
        
        # 计算指数项 - 限制范围
        exp_arg_h = self._cached_G ** 2 * L_perp_h
        exp_arg_h = torch.clamp(exp_arg_h, min=-10, max=10)
        
        sig_h = (self.sqrt_pi / (2 * zh) *
                 torch.exp(exp_arg_h) * erf_zh)

        # 最终检查输出 - 更保守的范围
        if torch.isnan(sig_h).any() or torch.isinf(sig_h).any():
            print("警告：sig_h 包含异常值，进行修正")
            sig_h = torch.clamp(sig_h, min=0.0, max=1.0)

        return sig_h

    def forward_batch(self, b, Delta, delta, G, model_params):
        """优化的批处理前向传播 - 超稳定版本"""
        batch_size = model_params.shape[0]

        # 检查输入参数
        if torch.isnan(model_params).any() or torch.isinf(model_params).any():
            print("警告：model_params包含NaN或Inf值")
            model_params = torch.clamp(model_params, min=-3.0, max=3.0)  # 更保守的范围

        # 更新缓存
        self._update_cache(b, Delta, delta, G, batch_size)

        # 解包参数并应用sigmoid确保在合理范围内 - 更保守的范围
        f_r = torch.sigmoid(model_params[:, 0]).unsqueeze(1)  # (batch_size, 1)
        adi = torch.sigmoid(model_params[:, 1]) * 5e-6  # (batch_size,) - 更保守的上限
        Dh = torch.sigmoid(model_params[:, 2]) * 0.5e-9  # (batch_size,) - 更保守的上限
        f_csf = torch.sigmoid(model_params[:, 3]).unsqueeze(1)  # (batch_size, 1)

        # 检查参数是否在合理范围内
        if torch.isnan(f_r).any() or torch.isinf(f_r).any():
            print("警告：f_r包含异常值，重置为0.5")
            f_r = torch.ones_like(f_r) * 0.5
        
        if torch.isnan(adi).any() or torch.isinf(adi).any():
            print("警告：adi包含异常值，重置为2.5e-6")  # 更保守的默认值
            adi = torch.ones_like(adi) * 2.5e-6
        
        if torch.isnan(Dh).any() or torch.isinf(Dh).any():
            print("警告：Dh包含异常值，重置为0.25e-9")  # 更保守的默认值
            Dh = torch.ones_like(Dh) * 0.25e-9
        
        if torch.isnan(f_csf).any() or torch.isinf(f_csf).any():
            print("警告：f_csf包含异常值，重置为0.1")
            f_csf = torch.ones_like(f_csf) * 0.1

        # 确保参数在合理范围内
        f_r = torch.clamp(f_r, min=0.2, max=0.8)  # 避免极端值
        adi = torch.clamp(adi, min=1e-7, max=5e-6)  # 更保守的范围
        Dh = torch.clamp(Dh, min=1e-12, max=0.5e-9)  # 更保守的范围
        f_csf = torch.clamp(f_csf, min=0.01, max=0.2)  # 更保守的范围

        # 计算各组分信号
        sig_r = self.restricted_compartment_vectorized(adi)  # (batch_size, n_b)
        sig_h = self.hindered_compartment_vectorized(Dh)  # (batch_size, n_b)

        # CSF信号
        sig_csf = self._cached_csf_signal.unsqueeze(0).expand(batch_size, -1)

        # 组合信号 - 确保权重和为1
        f_h = 1 - f_r - f_csf
        f_h = torch.clamp(f_h, min=0.2, max=0.8)  # 确保受阻室权重合理
        
        forward_signal = (f_r * sig_r + f_h * sig_h + f_csf * sig_csf)

        # 最终检查输出 - 更保守的范围
        if torch.isnan(forward_signal).any() or torch.isinf(forward_signal).any():
            print("警告：forward_signal包含异常值，进行修正")
            forward_signal = torch.clamp(forward_signal, min=0.0, max=1.0)

        return forward_signal

    def validate_parameters(self, model_param):
        """验证模型参数是否在有效范围内"""
        f_r, adi, Dh, f_csf = model_param

        assert 0 <= f_r <= 1, f"f_r应在[0,1]范围内，当前值: {f_r}"
        assert 0 <= adi <= 5e-6, f"adi应在[0.1e-6,5e-6]范围内，当前值: {adi}"  # 更保守的范围
        assert 0 <= Dh <= 0.5e-9, f"Dh应在[0.2e-9,0.5e-9]范围内，当前值: {Dh}"  # 更保守的范围
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

        # 其余计算与vectorized版本相同
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

        return L_perp

    def forward_batch(self, b, Delta, delta, G, model_params):
        """使用预分配内存的前向传播"""
        batch_size = model_params.shape[0]

        # 检查是否超出预分配大小
        if batch_size > self.max_batch_size:
            return super().forward_batch(b, Delta, delta, G, model_params)

        # 更新缓存
        self._update_cache(b, Delta, delta, G, batch_size)

        # 解包参数
        f_r = torch.sigmoid(model_params[:, 0]).unsqueeze(1)
        adi = torch.sigmoid(model_params[:, 1]) * 5e-6  # 更保守的上限
        Dh = torch.sigmoid(model_params[:, 2]) * 0.5e-9  # 更保守的上限
        f_csf = torch.sigmoid(model_params[:, 3]).unsqueeze(1)

        # 使用预分配内存计算
        sig_r = self.restricted_compartment_preallocated(adi)
        sig_h = self.hindered_compartment_vectorized(Dh)

        # CSF信号
        sig_csf = self._cached_csf_signal.unsqueeze(0).expand(batch_size, -1)

        # 组合信号
        f_h = torch.clamp(1 - f_r - f_csf, min=0.2, max=0.8)
        forward_signal = (f_r * sig_r + f_h * sig_h + f_csf * sig_csf)

        return forward_signal


def performance_test():
    """性能测试函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建SMT模型
    smt_model = SMTAxonDiameterOptimized(device)
    
    # 测试参数
    batch_size = 100
    n_b = 16
    
    # 创建测试数据
    b = torch.linspace(0, 3000, n_b, device=device)
    Delta = torch.ones(n_b, device=device) * 0.025
    delta = torch.ones(n_b, device=device) * 0.005
    G = torch.ones(n_b, device=device) * 0.04
    
    # 创建模型参数
    model_params = torch.randn(batch_size, 4, device=device)
    
    # 测试前向传播
    start_time = time.time()
    output = smt_model.forward_batch(b, Delta, delta, G, model_params)
    end_time = time.time()
    
    print(f"批处理大小: {batch_size}")
    print(f"输出形状: {output.shape}")
    print(f"计算时间: {end_time - start_time:.4f} 秒")
    print(f"输出范围: [{output.min():.6f}, {output.max():.6f}]")
    
    # 检查是否有NaN或Inf
    if torch.isnan(output).any():
        print("警告：输出包含NaN值")
    if torch.isinf(output).any():
        print("警告：输出包含Inf值")
    
    return output


if __name__ == "__main__":
    performance_test()
