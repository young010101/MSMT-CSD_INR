# forward model of the axon diameter in Python

import numpy as np 
import math
from scipy.special import erf
import matplotlib.pyplot as plt

gmr = 2.67e8

def smt_axon_diameter(b, Delta, delta, G, model_param):

    # model_param = [f_r, adi, Dh, f_csf]
    f_r   = model_param[0]
    adi   = model_param[1]
    Dh    = model_param[2]
    f_csf = model_param[3]

    D_csf = 3e-9 # the diffusivity of CSF is fixed

    sig_r = restricted_compartment(G, Delta, delta, adi)
    sig_h = hindered_compartment(G, Delta, delta, Dh)
    sig_csf = csf_compartment(b)

    forward_signal = f_r * sig_r + (1 - f_r - f_csf) * sig_h + f_csf * sig_csf

    return forward_signal


def restricted_compartment(G, Delta, delta, adi):

    D_r = 1.7e-9
    pi = math.pi

    bessel_root = np.array([1.84118378134066, 5.33144277352503, 8.53631636634629, 11.7060049025921, 14.8635886339090, 
                            18.0155278626818, 21.1643698591888, 24.3113268572108, 27.4570505710592, 30.6019229726691, 
                            33.7461828986674, 36.8899874092368, 40.0334440533507, 43.1766289654488, 46.3195975611739, 
                            49.4623911397028, 52.6050411115567, 55.7475717922510, 58.8900022991857, 62.0323478706620])

    E0_tmp = np.zeros([np.size(bessel_root), np.shape(Delta)[1]])

    alphm = bessel_root/((adi)/2)

    for ii in range(0, np.size(alphm)): 

        alpha_2 = alphm[ii]**2
        factor_1 = 2 * D_r * alpha_2 * delta
        factor_2 = 2 * np.exp(-D_r * alpha_2 * delta)
        factor_3 = 2 * np.exp(-D_r * alpha_2 * Delta)
        factor_4 = np.exp(-D_r * alpha_2 * (Delta-delta))
        factor_5 = np.exp(-D_r * alpha_2 * (Delta+delta))
        factor_6 = D_r**2 * alphm[ii]**6 * (((adi)/2)**2 * alpha_2 - 1)

        E0_tmp[ii, :] = (factor_1 - 2 + factor_2 + factor_3 - factor_4 - factor_5) / factor_6 

    E0_tmp_sum = np.sum(E0_tmp, axis=0)
    E0_tmp_sum = E0_tmp_sum.reshape([1, np.shape(Delta)[1]])

    L_perp = -2 * gmr**2 * E0_tmp_sum

    L_para = -(Delta - delta/3) * (gmr * delta)**2  * D_r

    zr = G * np.sqrt(L_perp - L_para)
    sig_r = np.sqrt(pi)/(2*zr) * np.exp(G**2 * L_perp) * erf(zr)

    return sig_r


def hindered_compartment(G, Delta, delta, Dh):
    D_r = 1.7e-9
    pi = math.pi

    L_para = ((delta/3-Delta)*(gmr*delta)**2)*D_r
    L_perp_h = ((delta/3-Delta)*(gmr*delta)**2)*Dh
    zh = G * np.sqrt(L_perp_h-L_para)
    sig_h = np.sqrt(pi)/(2 * zh)*np.exp(G**2 * L_perp_h) * erf(zh)

    return sig_h



def csf_compartment(b):
    
    D_csf = 3e-9
    sig_csf = np.exp(-b*D_csf)

    return sig_csf


if __name__ == "__main__":

    # [0, 1]
    # [0, 20e-6]
    # [0, 1.7e-9]
    # [0, 1]
    model_param = [0.5, 10e-6, 1.5e-9, 0.2]

    Delta = np.concatenate((np.ones([1,8])*19e-3, np.ones([1,8])*49e-3), axis=1)
    delta = np.concatenate((np.ones([1,8])*8e-3,  np.ones([1,8])*8e-3),  axis=1)
    bvals = np.array([50, 350, 800, 1500, 2400, 3450, 4750, 6000, 200, 950, 2300, 4250, 6750, 9850, 13500, 17800])* 1e6
    G = 1./(gmr * delta) * np.sqrt(bvals/(Delta-delta/3))
    q = (1/ 2 /math.pi) * gmr * G * delta

    sig = smt_axon_diameter(bvals, Delta, delta, G, model_param)

    plt.plot(np.arange(1, 9), sig[0, 0:8], color='blue', marker='o')
    plt.plot(np.arange(1, 9), sig[0, 8:], color='red', marker='s')

    plt.grid(True)
    plt.show()
