import math

import numpy as np
from experiments.reproduce_exp import exp_1, exp_4, exp_5, exp_6
import os
#Figure selection


figure = "Figure 1"
#code_path = os.path.dirname(os.path.realpath('__file__'))
#path      = os.path.join(code_path, 'Data')
# Reproduce Figure 1
if figure == "Figure 1":
    N = 10000
    dim_NN = 100
    dim_RF = 100
    dim_oracle = 100
    exp_1(N,dim_NN,dim_RF,dim_oracle)
elif figure == "Figure 4":
    K_arr = [4, 6, 8, 10, 12, 14, 16]  # array of number hidden neurons
    dim = 800
    N = 100000
    lr_NN = 0.01
    reg_NN = 0.1
    nr_sim_total = 20
    exp_4(N,K_arr,dim,N,lr_NN,reg_NN,nr_sim_total)
elif figure == "Figure 5":
    N = 400000
    dim_RF = [1000, 300, 400]
    gamma = [2, 5, 8]
    P = [2 * 1000, 5 * 300, 8 * 400]  # gamma*dim_RF
    num_sigmas = 15 * 3
    sigma1 = np.logspace(-2, -1, num=int(num_sigmas / 3))
    sigma2 = np.logspace(-1, 0, num=int(num_sigmas / 3))
    sigma3 = np.logspace(0, 1, num=int(num_sigmas / 3))
    sigma = np.round(np.append(sigma1, np.append(sigma2, sigma3)), 5)
    reg_RF = 0.0
    lr = 0.1
    exp_5(N,dim_RF,gamma,P,sigma,reg_RF,lr)

elif figure == "Figure 6":
    K = 10
    std_weights = 1e-2
    sigma = math.sqrt(0.05)
    dim1 = 200
    dim2 = 400
    num_t = 10 * 3
    t1 = np.logspace(1, 1.5, num=int(num_t / 3))
    t2 = np.logspace(1.5, 2, num=int(num_t / 3))
    t3 = np.logspace(2.5, 3, num=int(num_t / 3))
    t = np.round(np.append(t1, np.append(t2, t3)), 5)
    P1 = 2 * dim1
    P2 = 2 * dim2
    lr = 0.1
    reg = 0.0
    N1 = np.floor(t) * dim1
    N2 = np.floor(t) * dim2
    exp_6(t, N1, N2, P1, P2, K, dim1, dim2, lr, reg, sigma)
