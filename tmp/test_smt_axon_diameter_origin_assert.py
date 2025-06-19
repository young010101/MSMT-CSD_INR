# forward model of the axon diameter in Python

from src import smt_axon_diameter
import numpy as np
import math
import matplotlib.pyplot as plt



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
