import numpy as np
import torch
from dipy.data import get_sphere
from scipy.special import sph_harm
from utils import (
    cartesian_to_spherical,
    spherical_to_cartesian,
)


def create_y_mat(thetas: np.array, phis: np.array, l_max: int) -> torch.Tensor:
    n_dir = thetas.shape[0]
    n = (l_max + 1) * (l_max + 2) // 2

    y_mat = torch.zeros((n_dir, n))
    for l in range(0, l_max + 1, 2):
        for m in range(-l, l + 1):
            coef_idx = (l ** 2 + l) // 2 + m
            Y = sph_harm(np.abs(m), l, thetas, phis)
            if m < 0:
                y_mat[:, coef_idx] = torch.tensor(Y.imag * np.sqrt(2))
            if m > 0:
                y_mat[:, coef_idx] = torch.tensor(Y.real * np.sqrt(2))
            if m == 0:
                y_mat[:, coef_idx] = torch.tensor(Y.real)

    return y_mat


def get_rescale_value(l_max: int, rescale: bool = True) -> torch.Tensor:
    rescale_value = (
        torch.sqrt(4 * np.pi / (2 * torch.arange(0, l_max + 1, 2) + 1))
        if rescale
        else torch.ones(l_max // 2 + 1)
    )
    return rescale_value


def create_conv_vec(
        l_max: int, resp_coeff: torch.Tensor, rescale: bool = True
) -> torch.Tensor:
    rescale_value = get_rescale_value(l_max, rescale)
    conv_vec = torch.zeros((l_max + 1) * (l_max + 2) // 2)
    for i, l in enumerate(range(0, l_max + 1, 2)):
        for m in range(-l, l + 1):
            coef_idx = (l ** 2 + l) // 2 + m
            conv_vec[coef_idx] = resp_coeff[i] * rescale_value[i]
    return conv_vec


class SignalSingleShell:
    def __init__(
            self,
            l_max: int,
            resp_coeff: torch.Tensor,
            cart_bvec: np.array = None,
            sph_bvec: np.array = None,
            device: str = "cpu",
            fod_rescale: bool = True,
    ):
        self.l_max = l_max
        self.device = device

        self.y_mat = None

        sphere = get_sphere("repulsion724")  # 100, 200, 724
        sph_sphere_vecs = cartesian_to_spherical(sphere.vertices)
        sphere_thetas = sph_sphere_vecs[:, 2]
        sphere_phis = sph_sphere_vecs[:, 1]
        self.sphere_y_mat = create_y_mat(sphere_thetas, sphere_phis, l_max).to(device)

        if cart_bvec is not None:
            self.cart_bvec = cart_bvec
            sph_bvec = cartesian_to_spherical(cart_bvec)

        if sph_bvec is not None:
            self.sph_bvec = sph_bvec

            if cart_bvec is None:
                self.cart_bvec = spherical_to_cartesian(sph_bvec)

            sample_phis = sph_bvec[..., 1]
            sample_thetas = sph_bvec[..., 2]
            self.y_mat = create_y_mat(sample_thetas, sample_phis, l_max).to(device)

        self.conv_vec = create_conv_vec(l_max, resp_coeff, fod_rescale).to(device)

    def compute_signal_from_coeff(self, coeffs: torch.tensor):
        return torch.einsum("bk, k, dk -> bd", coeffs, self.conv_vec, self.y_mat)

    def compute_negative_signal(self, coeffs: torch.Tensor):
        # return torch.clamp(
        #     torch.einsum("bk, dk -> bd", coeffs, self.sphere_y_mat), max=0
        # )

        amplitudes = torch.einsum("bk, dk -> bd", coeffs, self.sphere_y_mat)
        max_values = torch.mean(amplitudes, dim=1, keepdim=True) * 0.1
        return torch.clamp(amplitudes - max_values, max=0)


class SignalMultishell:
    def __init__(
            self,
            l_max: int,
            resp_coeff: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            bval_idx: list,
            cart_bvec: np.array = None,
            sph_bvec: np.array = None,
            device: str = "cpu",
            fod_rescale: bool = True,
    ):
        self.l_max = l_max
        self.device = device

        self.y_mat = None
        wm_coeff, gm_coeff, csf_coeff = (
            resp_coeff[0],
            resp_coeff[1].to(device),
            resp_coeff[2].to(device),
        )

        self.gm_coeff = torch.stack([gm_coeff[idx] for idx in bval_idx])
        self.csf_coeff = torch.stack([csf_coeff[idx] for idx in bval_idx])

        conv_vecs = [
            create_conv_vec(l_max, coeffs, fod_rescale).to(device)
            for coeffs in wm_coeff
        ]

        self.conv_mat = torch.stack([conv_vecs[idx] for idx in bval_idx])
        self.fod_rescale = fod_rescale

        sphere = get_sphere("repulsion724")  # 100, 200, 724
        sph_sphere_vecs = cartesian_to_spherical(sphere.vertices)
        sphere_thetas = sph_sphere_vecs[:, 2]
        sphere_phis = sph_sphere_vecs[:, 1]
        self.sphere_y_mat = create_y_mat(sphere_thetas, sphere_phis, l_max).to(device)

        if cart_bvec is not None:
            self.cart_bvec = cart_bvec
            sph_bvec = cartesian_to_spherical(cart_bvec)

        if sph_bvec is not None:
            self.sph_bvec = sph_bvec
            if cart_bvec is None:
                self.cart_bvec = spherical_to_cartesian(sph_bvec)

            sample_phis = sph_bvec[..., 1]
            sample_thetas = sph_bvec[..., 2]
            self.y_mat = create_y_mat(sample_thetas, sample_phis, l_max).to(device)

    def compute_signal_from_coeff(self, coeffs: torch.tensor):
        fod_coeffs = coeffs[:, :-2]
        # fod_frac = coeffs[:, -3]
        gm_coeff = coeffs[:, [-2]]
        csf_coeff = coeffs[:, [-1]]

        fod_signal = torch.einsum(
            "bk, dk, dk -> bd", fod_coeffs, self.conv_mat, self.y_mat
        )  # * fod_frac[:, None, None]

        gm_signal = torch.einsum(
            "bk, dk -> bd", gm_coeff, self.gm_coeff
        )
        csf_signal = torch.einsum(
            "bk, dk -> bd", csf_coeff, self.csf_coeff
        )

        return fod_signal + gm_signal + csf_signal

    def compute_negative_signal(self, coeffs: torch.Tensor):
        fod_coeffs = coeffs[:, :-2]
        amplitudes = torch.einsum("bk, dk -> bd", fod_coeffs, self.sphere_y_mat)
        max_values = torch.mean(amplitudes, dim=1, keepdim=True) * 0.1
        return torch.clamp(amplitudes - max_values, max=0)


# class VonShell:
#     def __init__(
#             self,
#             device: str = "cpu",
#     ):
#         # 将常量移到初始化中
#         self.gmr = 2.67e8
#         self._Delta = torch.cat((torch.ones(1, 8) * 19e-3, torch.ones(1, 8) * 49e-3), dim=1)
#         self._Delta = self._Delta.to(device)
#         self._delta = torch.cat((torch.ones(1, 8) * 8e-3, torch.ones(1, 8) * 8e-3), dim=1)
#         self._delta = self._delta.to(device)
#         self._bvals = torch.tensor([50, 350, 800, 1500, 2400, 3450, 4750, 6000,
#                                     200, 950, 2300, 4250, 6750, 9850, 13500, 17800],
#                                    dtype=torch.float64) * 1e6
#         self._bvals = self._bvals.to(device)
#         G = 1.0 / (self.gmr * self._delta) * torch.sqrt(self._bvals / (self._Delta - self._delta / 3))
#         self.G = G
#
#     def compute_signal_from_coeff(self, coeffs: torch.tensor):
#         # 使用批处理版本
#         sig_tensor = smt_axon_diameter_batch(self._bvals, self._Delta, self._delta, self.G, coeffs)
#         return sig_tensor
#
#     def compute_negative_signal(self, coeffs: torch.Tensor):
#         return torch.zeros((1, 1))


class VonShell:
    def __init__(
            self,
            device: str = "cpu",
            max_batch_size: int = 1000,  # 添加最大批次大小参数
    ):
        self.device = device
        self.max_batch_size = max_batch_size

        # 将常量移到初始化中并移到指定设备
        self.gmr = 2.67e8
        self._Delta = torch.cat((torch.ones(1, 8) * 19e-3, torch.ones(1, 8) * 49e-3), dim=1).to(device)
        self._delta = torch.cat((torch.ones(1, 8) * 8e-3, torch.ones(1, 8) * 8e-3), dim=1).to(device)
        self._bvals = torch.tensor([50, 350, 800, 1500, 2400, 3450, 4750, 6000,
                                    200, 950, 2300, 4250, 6750, 9850, 13500, 17800],
                                   dtype=torch.float64, device=device) * 1e6

        # 预计算G
        self.G = 1.0 / (self.gmr * self._delta) * torch.sqrt(self._bvals / (self._Delta - self._delta / 3))

        # 初始化优化的SMT模型
        from src import SMTAxonDiameterOptimized
        self.smt_model = SMTAxonDiameterOptimized(device, max_batch_size)

    def compute_signal_from_coeff(self, coeffs: torch.tensor):
        """
        计算信号，支持批处理
        """
        # 检查输入是否包含 NaN
        if torch.isnan(coeffs).any():
            print("警告：输入 coeffs 包含 NaN 值")
            # 将 NaN 替换为安全值
            coeffs = torch.where(torch.isnan(coeffs), torch.zeros_like(coeffs), coeffs)

        # 应用 sigmoid 函数确保所有系数都在 (0, 1) 范围内
        normalized_coeffs = torch.sigmoid(coeffs)

        # 再次检查 sigmoid 后是否有异常值
        if torch.isnan(normalized_coeffs).any() or torch.isinf(normalized_coeffs).any():
            print("警告：sigmoid 后出现 NaN 或 Inf")
            normalized_coeffs = torch.clamp(normalized_coeffs, min=1e-8, max=1 - 1e-8)

        # 使用归一化后的系数进行计算
        try:
            sig_tensor = self.smt_model.forward_batch(
                self._bvals, self._Delta, self._delta, self.G, normalized_coeffs
            )

            # 检查输出
            if torch.isnan(sig_tensor).any() or torch.isinf(sig_tensor).any():
                print("警告：SMT 模型输出包含 NaN 或 Inf")
                sig_tensor = torch.where(
                    torch.isnan(sig_tensor) | torch.isinf(sig_tensor),
                    torch.zeros_like(sig_tensor),
                    sig_tensor
                )

            return sig_tensor

        except Exception as e:
            print(f"SMT 计算错误: {e}")
            # 返回安全的默认值
            batch_size = coeffs.shape[0]
            n_directions = self._bvals.shape[0]
            return torch.zeros((batch_size, n_directions), device=self.device)

    def compute_negative_signal(self, coeffs: torch.Tensor):
        """
        计算负值约束项用于正则化

        对于SMT模型的4个参数，我们施加以下约束：
        1. 所有参数都应该在合理的物理范围内
        2. 惩罚过小或过大的值
        """
        # # 应用sigmoid确保参数在(0,1)范围内后，计算偏离合理范围的惩罚
        # normalized_coeffs = torch.sigmoid(coeffs)
        #
        # # 对于SMT模型，参数应该在中等范围内（例如0.1-0.9）
        # # 惩罚过于极端的值
        # lower_bound = 0.1
        # upper_bound = 0.9
        #
        # # 计算低于下界的惩罚
        # lower_penalty = torch.clamp(lower_bound - normalized_coeffs, min=0) ** 2
        #
        # # 计算高于上界的惩罚
        # upper_penalty = torch.clamp(normalized_coeffs - upper_bound, min=0) ** 2
        #
        # # 返回总惩罚，保持与其他类一致的形状 (batch_size, n_directions)
        # total_penalty = torch.sum(lower_penalty + upper_penalty, dim=-1, keepdim=True)
        #
        # # 扩展到匹配b值方向数
        # n_directions = self._bvals.shape[0]
        # return total_penalty.expand(-1, n_directions)
        """
        改进的负值约束，特别针对adi参数
        """
        sigmoid_coeffs = torch.sigmoid(coeffs)

        # 特别惩罚adi参数过小的情况
        adi_normalized = sigmoid_coeffs[:, 1]  # adi参数
        adi_penalty = torch.relu(0.005 - adi_normalized) ** 2  # 惩罚小于0.005的值

        # 其他常规约束
        f_r = sigmoid_coeffs[:, 0]
        f_csf = sigmoid_coeffs[:, 3]
        fraction_penalty = torch.relu(f_r + f_csf - 0.95) ** 2

        # 原始系数极端值惩罚
        extreme_penalty = torch.sum(
            torch.relu(torch.abs(coeffs) - 3.0) ** 2,
            dim=-1
        )

        total_penalty = (adi_penalty + fraction_penalty + extreme_penalty).unsqueeze(1)

        # 扩展到匹配输出维度
        n_directions = self._bvals.shape[0]
        return total_penalty.expand(-1, n_directions)

    def reset_cache(self):
        """重置SMT模型的缓存（在改变批次大小时可能需要）"""
        if hasattr(self.smt_model, '_reset_cache'):
            self.smt_model._reset_cache()

    def get_parameter_bounds(self):
        """
        返回SMT模型参数的合理边界

        Returns:
            dict: 包含参数边界的字典
        """
        return {
            'f_r': (0.0, 1.0),  # 受限分数
            'adi': (0.05, 1.0),  # 轴突直径指数 (实际值会乘以20e-6)
            'Dh': (0.2 / 1.7, 1.0),  # 受阻扩散系数 (实际值会乘以1.7e-9)
            'f_csf': (0.0, 1.0),  # CSF分数
        }

    def validate_coeffs(self, coeffs: torch.Tensor):
        """
        验证输入系数的有效性

        Args:
            coeffs: 形状为 (batch_size, 4) 的系数张量

        Returns:
            bool: 系数是否有效
        """
        if coeffs.shape[1] != 4:
            return False

        bounds = self.get_parameter_bounds()

        # 检查f_r边界
        if torch.any(coeffs[:, 0] < bounds['f_r'][0]) or torch.any(coeffs[:, 0] > bounds['f_r'][1]):
            return False

        # 检查adi边界
        if torch.any(coeffs[:, 1] < bounds['adi'][0]) or torch.any(coeffs[:, 1] > bounds['adi'][1]):
            return False

        # 检查Dh边界
        if torch.any(coeffs[:, 2] < bounds['Dh'][0]) or torch.any(coeffs[:, 2] > bounds['Dh'][1]):
            return False

        # 检查f_csf边界
        if torch.any(coeffs[:, 3] < bounds['f_csf'][0]) or torch.any(coeffs[:, 3] > bounds['f_csf'][1]):
            return False

        # 检查分数和约束: f_r + f_csf <= 1
        if torch.any(coeffs[:, 0] + coeffs[:, 3] > 1.0):
            return False

        return True