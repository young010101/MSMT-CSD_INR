import torch
import math
from scipy.special import erf
import matplotlib.pyplot as plt

# Constants
gmr = 2.67e8


def smt_axon_diameter(b, Delta, delta, G, model_param):
    f_r, adi, Dh, f_csf = model_param

    sig_r = restricted_compartment(G, Delta, delta, adi)
    sig_h = hindered_compartment(G, Delta, delta, Dh)
    sig_csf = csf_compartment(b)

    forward_signal = f_r * sig_r + (1 - f_r - f_csf) * sig_h + f_csf * sig_csf
    return forward_signal


def restricted_compartment(G, Delta, delta, adi):
    D_r = 1.7e-9
    pi = math.pi

    bessel_root = torch.tensor([
        1.84118378134066, 5.33144277352503, 8.53631636634629, 11.7060049025921, 14.8635886339090,
        18.0155278626818, 21.1643698591888, 24.3113268572108, 27.4570505710592, 30.6019229726691,
        33.7461828986674, 36.8899874092368, 40.0334440533507, 43.1766289654488, 46.3195975611739,
        49.4623911397028, 52.6050411115567, 55.7475717922510, 58.8900022991857, 62.0323478706620
    ])

    alphm = bessel_root / (adi / 2)
    n_b = Delta.shape[1]
    E0_tmp = torch.zeros(len(alphm), n_b)

    for ii, alpha in enumerate(alphm):
        alpha_2 = alpha ** 2
        factor_1 = 2 * D_r * alpha_2 * delta
        factor_2 = 2 * torch.exp(-D_r * alpha_2 * delta)
        factor_3 = 2 * torch.exp(-D_r * alpha_2 * Delta)
        factor_4 = torch.exp(-D_r * alpha_2 * (Delta - delta))
        factor_5 = torch.exp(-D_r * alpha_2 * (Delta + delta))
        factor_6 = D_r**2 * alpha**6 * ((adi / 2)**2 * alpha_2 - 1)

        E0_tmp[ii, :] = (factor_1 - 2 + factor_2 + factor_3 - factor_4 - factor_5) / factor_6

    E0_tmp_sum = E0_tmp.sum(dim=0, keepdim=True)
    L_perp = -2 * gmr**2 * E0_tmp_sum
    L_para = -(Delta - delta / 3) * (gmr * delta)**2 * D_r

    zr = G * torch.sqrt(L_perp - L_para)
    sig_r = torch.sqrt(torch.tensor(pi)) / (2 * zr) * torch.exp(G**2 * L_perp) * torch.tensor(erf(zr))
    return sig_r


def hindered_compartment(G, Delta, delta, Dh):
    D_r = 1.7e-9
    pi = math.pi

    L_para = ((delta / 3 - Delta) * (gmr * delta)**2) * D_r
    L_perp_h = ((delta / 3 - Delta) * (gmr * delta)**2) * Dh

    zh = G * torch.sqrt(L_perp_h - L_para)
    sig_h = torch.sqrt(torch.tensor(pi)) / (2 * zh) * torch.exp(G**2 * L_perp_h) * torch.tensor(erf(zh))
    return sig_h


def csf_compartment(b):
    D_csf = 3e-9
    return torch.exp(-b * D_csf)


if __name__ == "__main__":
    model_param = [0.5, 10e-6, 1.5e-9, 0.2]

    Delta = torch.cat((torch.ones(1, 8) * 19e-3, torch.ones(1, 8) * 49e-3), dim=1)
    delta = torch.cat((torch.ones(1, 8) * 8e-3, torch.ones(1, 8) * 8e-3), dim=1)
    bvals = torch.tensor([50, 350, 800, 1500, 2400, 3450, 4750, 6000,
                          200, 950, 2300, 4250, 6750, 9850, 13500, 17800], dtype=torch.float64) * 1e6

    G = 1.0 / (gmr * delta) * torch.sqrt(bvals / (Delta - delta / 3))

    sig = smt_axon_diameter(bvals, Delta, delta, G, model_param)

    plt.plot(range(1, 9), sig[0, 0:8].numpy(), color='blue', marker='o')
    plt.plot(range(1, 9), sig[0, 8:].numpy(), color='red', marker='s')
    plt.grid(True)
    plt.savefig(f"axon_diameter/1.png")
    plt.show()