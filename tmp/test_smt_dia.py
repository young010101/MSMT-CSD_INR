import torch
import math
from torch.special import erf

# 全局常量，避免重复计算
gmr = 2.67e8
D_r = 1.7e-9
D_csf = 3e-9

# 预计算贝塞尔根（移到全局）
BESSEL_ROOTS = torch.tensor([
    1.84118378134066, 5.33144277352503, 8.53631636634629, 11.7060049025921, 14.8635886339090,
    18.0155278626818, 21.1643698591888, 24.3113268572108, 27.4570505710592, 30.6019229726691,
    33.7461828986674, 36.8899874092368, 40.0334440533507, 43.1766289654488, 46.3195975611739,
    49.4623911397028, 52.6050411115567, 55.7475717922510, 58.8900022991857, 62.0323478706620
])


def smt_axon_diameter_batch_optimized(b, Delta, delta, G, model_params):
    """
    完全向量化的批处理版本
    """
    device = model_params.device
    batch_size = model_params.shape[0]

    # 解包参数
    f_r = model_params[:, 0:1]  # (batch_size, 1)
    adi = model_params[:, 1:2]  # (batch_size, 1)
    Dh = model_params[:, 2:3]  # (batch_size, 1)
    f_csf = model_params[:, 3:4]  # (batch_size, 1)

    # 预计算常量
    bessel_root = BESSEL_ROOTS.to(device)
    sqrt_pi = math.sqrt(math.pi)

    # restricted compartment - 完全向量化
    sig_r = restricted_compartment_vectorized(G, Delta, delta, adi, bessel_root, device)

    # hindered compartment - 向量化
    sig_h = hindered_compartment_vectorized(G, Delta, delta, Dh, device)

    # CSF compartment
    sig_csf = torch.exp(-b * D_csf).unsqueeze(0).expand(batch_size, -1)

    # 最终信号计算
    forward_signal = f_r * sig_r + (1 - f_r - f_csf) * sig_h + f_csf * sig_csf
    return forward_signal


def restricted_compartment_vectorized(G, Delta, delta, adi, bessel_root, device):
    """
    完全向量化的restricted compartment计算
    """
    batch_size = adi.shape[0]
    n_b = Delta.shape[1]
    n_roots = len(bessel_root)

    # 计算alphm: (batch_size, n_roots)
    alphm = bessel_root.unsqueeze(0) / (adi / 2)  # broadcast

    # 向量化计算所有factor - 关键优化点
    alpha_2 = alphm.unsqueeze(2) ** 2  # (batch_size, n_roots, 1)

    # 预计算delta和Delta相关项
    delta_exp = torch.exp(-D_r * alpha_2 * delta.unsqueeze(0))  # (batch_size, n_roots, n_b)
    Delta_exp = torch.exp(-D_r * alpha_2 * Delta.unsqueeze(0))

    # 一次性计算所有factor
    factor_1 = 2 * D_r * alpha_2 * delta.unsqueeze(0)
    factor_2 = 2 * delta_exp
    factor_3 = 2 * Delta_exp
    factor_4 = torch.exp(-D_r * alpha_2 * (Delta - delta).unsqueeze(0))
    factor_5 = torch.exp(-D_r * alpha_2 * (Delta + delta).unsqueeze(0))

    # 分母计算
    adi_expanded = adi.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1)
    factor_6 = D_r ** 2 * alphm.unsqueeze(2) ** 6 * ((adi_expanded / 2) ** 2 * alpha_2 - 1)

    # E0计算 - 向量化
    E0_tmp = (factor_1 - 2 + factor_2 + factor_3 - factor_4 - factor_5) / factor_6

    # 求和
    E0_tmp_sum = E0_tmp.sum(dim=1)  # (batch_size, n_b)

    # 最终计算
    L_perp = -2 * gmr ** 2 * E0_tmp_sum
    L_para = -(Delta - delta / 3) * (gmr * delta) ** 2 * D_r

    zr = G * torch.sqrt(L_perp - L_para)
    sig_r = math.sqrt(math.pi) / (2 * zr) * torch.exp(G ** 2 * L_perp) * erf(zr)

    return sig_r


def hindered_compartment_vectorized(G, Delta, delta, Dh, device):
    """向量化的hindered compartment"""
    L_para = ((delta / 3 - Delta) * (gmr * delta) ** 2) * D_r
    L_perp_h = ((delta / 3 - Delta) * (gmr * delta) ** 2) * Dh

    zh = G * torch.sqrt(L_perp_h - L_para)
    sig_h = math.sqrt(math.pi) / (2 * zh) * torch.exp(G ** 2 * L_perp_h) * erf(zh)
    return sig_h


# 缓存优化版本
class VonShellOptimized:
    def __init__(self):
        self.gmr = 2.67e8
        self._Delta = torch.cat((torch.ones(1, 8) * 19e-3, torch.ones(1, 8) * 49e-3), dim=1)
        self._delta = torch.cat((torch.ones(1, 8) * 8e-3, torch.ones(1, 8) * 8e-3), dim=1)
        self._bvals = torch.tensor([50, 350, 800, 1500, 2400, 3450, 4750, 6000,
                                    200, 950, 2300, 4250, 6750, 9850, 13500, 17800],
                                   dtype=torch.float64) * 1e6

        # 预计算G矩阵，避免重复计算
        self._G_cache = {}

    def _get_cached_params(self, device):
        """获取缓存的参数"""
        if device not in self._G_cache:
            Delta = self._Delta.to(device)
            delta = self._delta.to(device)
            bvals = self._bvals.to(device)
            G = 1.0 / (self.gmr * delta) * torch.sqrt(bvals / (Delta - delta / 3))

            self._G_cache[device] = {
                'Delta': Delta,
                'delta': delta,
                'bvals': bvals,
                'G': G
            }

        return self._G_cache[device]

    def compute_signal_from_coeff(self, coeffs: torch.tensor):
        device = coeffs.device
        params = self._get_cached_params(device)

        # 缩放系数 - 原地操作优化
        coeffs_scaled = coeffs.clone()
        coeffs_scaled[:, 1] *= 20e-6
        coeffs_scaled[:, 2] *= 1.7e-9

        # 使用优化版本
        sig_tensor = smt_axon_diameter_batch_optimized(
            params['bvals'], params['Delta'], params['delta'],
            params['G'], coeffs_scaled
        )
        return sig_tensor

    def compute_negative_signal(self, coeffs: torch.Tensor):
        return torch.zeros((coeffs.shape[0], 1), device=coeffs.device)




# 测试代码
def test_von_shell_optimized():
    """测试VonShellOptimized类的功能"""

    von = VonShellOptimized()

    # 创建测试数据
    batch_size = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 生成测试系数 (batch_size, 3)
    test_coeffs = torch.tensor([
        [0.5, 1.0, 1.5],
        [0.3, 1.2, 1.8],
        [0.7, 0.8, 1.2],
        [0.4, 1.5, 2.0],
        [0.6, 0.9, 1.6]
    ], dtype=torch.float64, device=device)

    # 测试信号计算
    print("测试 VonShellOptimized 类...")
    print(f"输入系数形状: {test_coeffs.shape}")
    print(f"设备: {device}")

    try:
        # 计算信号
        signal_result = von.compute_signal_from_coeff(test_coeffs)
        print(f"信号输出形状: {signal_result.shape}")
        print(f"信号值范围: [{signal_result.min().item():.6f}, {signal_result.max().item():.6f}]")

        # 测试负信号
        negative_signal = von.compute_negative_signal(test_coeffs)
        print(f"负信号输出形状: {negative_signal.shape}")
        print(f"负信号是否全为零: {torch.allclose(negative_signal, torch.zeros_like(negative_signal))}")

        # 验证缓存机制
        params1 = von._get_cached_params(device)
        params2 = von._get_cached_params(device)
        print(f"缓存机制工作正常: {params1 is params2}")

        print("所有测试通过!")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # test_von_shell_optimized()
    von = VonShellOptimized()
    model_param = torch.tensor([[0.5, 10e-6, 1.5e-9, 0.2]])
    signal = von.compute_signal_from_coeff(model_param)
    print(signal)
    print(signal.shape)
    # import matplotlib.pyplot as plt
    # plt.plot(signal[0, :])
    # plt.show()