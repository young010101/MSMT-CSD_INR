import math
from functools import lru_cache

import numpy as np
import torch
from dipy.data import get_sphere
from scipy.special import sph_harm
from utils import (
    cartesian_to_spherical,
    spherical_to_cartesian,
)

from smt_axon_diameter_tensor import smt_axon_diameter


def create_y_mat(thetas: np.array, phis: np.array, l_max: int) -> torch.Tensor:
    n_dir = thetas.shape[0]
    n = (l_max + 1) * (l_max + 2) // 2

    y_mat = torch.zeros((n_dir, n))
    for l in range(0, l_max + 1, 2):
        for m in range(-l, l + 1):
            coef_idx = (l**2 + l) // 2 + m
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
            coef_idx = (l**2 + l) // 2 + m
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


class VonShell:
    def compute_signal_from_coeff(self, coeffs: torch.tensor):
        device = coeffs.device
        gmr = 2.67e8
        
        Delta = torch.cat((torch.ones(1, 8) * 19e-3, torch.ones(1, 8) * 49e-3), dim=1).to(device)
        delta = torch.cat((torch.ones(1, 8) * 8e-3, torch.ones(1, 8) * 8e-3), dim=1).to(device)
        bvals = torch.tensor([50, 350, 800, 1500, 2400, 3450, 4750, 6000,
                          200, 950, 2300, 4250, 6750, 9850, 13500, 17800], dtype=torch.float64, device=device) * 1e6

        G = 1.0 / (gmr * delta) * torch.sqrt(bvals / (Delta - delta / 3))
        # Delta = np.concatenate((np.ones([1,8])*19e-3, np.ones([1,8])*49e-3), axis=1)
        # delta = np.concatenate((np.ones([1,8])*8e-3,  np.ones([1,8])*8e-3),  axis=1)
        # bvals = np.array([50, 350, 800, 1500, 2400, 3450, 4750, 6000, 200, 950, 2300, 4250, 6750, 9850, 13500, 17800])* 1e6
        # G = 1./(gmr * delta) * np.sqrt(bvals/(Delta-delta/3))
        model_param = coeffs
        # coeffs_np = coeffs.detach().cpu().numpy()
        # sig_np = smt_axon_diameter(bvals, Delta, delta, G, coeffs_np)
        # sig_tensor = torch.tensor(sig_np, device=coeffs.device, dtype=coeffs.dtype)
        # return sig
    
        # coeffs_np = coeffs.detach().cpu().numpy()
        coeffs_np = coeffs
        all_signals = []

        # [0, 1]
        # [0, 20e-6]
        # [0, 1.7e-9]
        # [0, 1]
        coeffs_np[:, 1] = coeffs_np[:, 1] * 20e-6
        coeffs_np[:, 2] = coeffs_np[:, 2] * 1.7e-9
        for i in range(coeffs_np.shape[0]):
            single_coeff = coeffs_np[i]  # shape (4,)
            sig_np = smt_axon_diameter(bvals, Delta, delta, G, single_coeff)  # shape (num_bvals,)
            all_signals.append(sig_np)

        # stack all signal outputs back into one tensor: shape (batch_size, num_bvals)
        sig_np_batch = torch.stack(all_signals, axis=0)
        # sig_tensor = torch.tensor(sig_np_batch, device=coeffs.device, dtype=coeffs.dtype, requires_grad=True)
        sig_tensor = sig_np_batch
        return sig_tensor
    def compute_negative_signal(self, coeffs: torch.Tensor):
        return torch.zeros((1,1))