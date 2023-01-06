import math

from matplotlib import pyplot as plt
import numpy as np
import os

from utils.utils import plot_input_feature_spaces


def plot_figure1(oracle_error,RF_error,NN_error,sigma,dim_RF):
    """
    Final plot for Figure 1. Before running this box, make sure to run everything in the following parts:
      1) Defining the functions frequently used in this notebook
      2) Figure 1 (only box 1 if you don't want to run the simulation yourself)
    """

    # These are the saved arrays. If you ran it yourself, please comment out the following lines.
    #NN_error      = np.load(os.path.join(path, 'NN_error_fig1.npy'))
    #oracle_error  = np.load(os.path.join(path, 'oracle_error_fig1.npy'))
    #RF_error      = np.load(os.path.join(path, 'RF_error_fig1.npy'))


    ######### Plot Figure 1 ##############################################
    ######################################################################

    y = np.ones((len(sigma)))*0.5
    plt.figure(figsize = (15, 7))
    plt.rcParams.update({'font.size': 16})
    plt.plot(sigma, y,            color='black', linestyle='dashed', label='random \nguessing')
    plt.plot(sigma, oracle_error, color='blue' , linestyle='solid',  label='oracle')
    plt.plot(sigma, RF_error,     color='red'  , linestyle='solid',  label='Random Features')
    plt.plot(sigma, NN_error,     color='green', linestyle='dashed', label='2LNN')
    plt.xscale('log')
    plt.ylim([-0.01, 0.55])
    plt.vlines(0.25, -0.01, 0.55, 'grey')
    plt.vlines(0.3, -0.01, 0.55, 'grey')
    plt.title('Random Features and 2LNN on high-dimensional Gaussian mixture classification\n\nHigh SNR                         Low SNR')
    plt.xlabel('Standard Deviation e.g. 1/SNR')
    plt.ylabel('Classification Error')
    plt.legend()
    plt.show()


    ######### Plot Input and RF Feature Space #############################
    ######################################################################

    # Comment: As F is a random transformation matrix, this plot will not
    # be the exact same one as shown ontop. Feel free to run this a couple
    # of times to get a figure as close as possible to the one displayed.

    plot_input_feature_spaces(dim_RF, 0.0001, mu_r_1=None, mu_r_2=None)


def plot_figure4(result_1,result_001,K_arr):
    """
    Final Plot for Figure 4. Before running this box, make sure to run everything in the following parts:
      1) Defining the functions frequently used in this notebook
      2) Figure 4 (only this box if you don't want to run the simulation again)
    """

    # These are the saved arrays. If you ran it yourself, please comment out the following lines.
    #K_arr = [4, 6, 8, 10, 12, 14, 16]
    #result_001 = np.load(os.path.join(path, 'result001_fig4.npy'))
    #result_1 = np.load(os.path.join(path, 'result1_fig4.npy'))

    ######### Plot Figure ################################################
    ######################################################################

    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 16})
    plt.ylim([0, 1.5])
    plt.plot(K_arr, result_001, '+', label='simga_0 = 0.01')
    plt.plot(K_arr, result_1, '+r', label='simga_0 = 1')
    plt.title(
        'Fraction of simulations that converged to the optimal solution \nfor the 2LNN out of 20 simulations for increasing values of K')
    plt.xlabel('number of hidden neurons K')
    plt.ylabel('N_converged/N_total')
    plt.ylim(-0.01, 1.1)
    plt.legend()
    plt.show()


def plot_figure5(num_sigmas,result_gamma_2_dim_1000,result_gamma_5_dim_300,result_gamma_8_dim_400):
    ######### Plot performances ##########################################
    ######################################################################
    """
    Final Plot for Figure 5. Before running this box, make sure to run everything in the following parts:
      1) Defining the functions frequently used in this notebook
      2) Figure 5 (only this box if you don't want to run the simulation yourself)
    """

    # These are the saved arrays. If you ran it yourself, please comment out the following lines.
    sigma1 = np.logspace(-2, -1, num=int(num_sigmas / 3))
    sigma2 = np.logspace(-1, 0, num=int(num_sigmas / 3))
    sigma3 = np.logspace(0, 1, num=int(num_sigmas / 3))
    sigma = np.round(np.append(sigma1, np.append(sigma2, sigma3)), 5)
    #result_gamma_8_dim_400 = np.load(os.path.join(path, 'result_gamma_8_dim_400.npy'))
    #result_gamma_5_dim_300 = np.load(os.path.join(path, 'result_gamma_5_dim_300.npy'))
    #result_gamma_2_dim_1000 = np.load(os.path.join(path, 'result_gamma_2_dim_1000.npy'))

    plt.figure(figsize=(15, 7))
    plt.rcParams.update({'font.size': 16})
    plt.plot(sigma, result_gamma_2_dim_1000, 'green', label="D = 1000, \ngamma = 2")
    plt.plot(sigma, result_gamma_5_dim_300, 'red', label="D = 300,  \ngamma = 5")
    plt.plot(sigma, result_gamma_8_dim_400, 'blue', label="D = 400,  \ngamma = 8")
    plt.legend(loc="lower right")
    plt.xscale('log')
    plt.ylim([0.0, 0.6])
    plt.title('Evolution of the classification error of RF for various sigma and gamma')
    plt.xlabel('( sigma*D^(1/4) )/gamma^(1/4)')
    plt.ylabel('classification error')
    plt.show()



def plot_figure6(t,dim1,dim2,mse_2LNN_200_low,mse_2LNN_400_low,mse_RF_200_low,mse_RF_400_low,mse_2LNN_200_high,mse_2LNN_400_high,
                 mse_RF_200_high, mse_RF_400_high, mse_2LNN_200_mixed, mse_2LNN_400_mixed, mse_RF_200_mixed, mse_RF_400_mixed,
                 ):
    """
    Final Plot for Figure 6. Before running this box, make sure to run everything in the following parts:
      1) Defining the functions frequently used in this notebook
      2) Figure 6 (only box 1 if you don't want to run the simulation yourself)
    """
    ######### Plot performances ##########################################
    ######################################################################
    plt.figure(figsize=(15, 15))
    plt.subplot(3, 2, 1)
    plt.plot(t, mse_2LNN_200_low, 'red', label="D = 200")
    plt.plot(t, mse_2LNN_400_low, 'black', label="D = 400")
    plt.legend(loc="lower right")
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-3, 1e0 + 0.5])
    plt.xlabel('t')
    plt.ylabel('mse 2LNN')

    plt.subplot(3, 2, 2)
    plt.plot(t, mse_RF_200_low, 'red', label="D = 200")
    plt.plot(t, mse_RF_400_low, 'black', label="D = 400")
    plt.legend(loc="lower right")
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-3, 1e0 + 0.5])
    plt.xlabel('t')
    plt.ylabel('mse RF')

    plt.subplot(3, 2, 3)
    plt.plot(np.sqrt(dim1) * t, mse_2LNN_200_high, 'red', label="D = 200")
    plt.plot(np.sqrt(dim2) * t, mse_2LNN_400_high, 'black', label="D = 400")
    plt.legend(loc="lower right")
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-3, 1e0 + 0.5])
    plt.xlabel('t√D')
    plt.ylabel('mse 2LNN')

    plt.subplot(3, 2, 4)
    plt.plot((dim1) * t, mse_RF_200_high, 'red', label="D = 200")
    plt.plot((dim2) * t, mse_RF_400_high, 'black', label="D = 400")
    plt.legend(loc="lower right")
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-3, 1e0 + 0.5])
    plt.xlabel('tD')
    plt.ylabel('mse RF')

    plt.subplot(3, 2, 5)
    plt.plot(t, mse_2LNN_200_mixed, 'red', label="D = 200")
    plt.plot(t, mse_2LNN_400_mixed, 'black', label="D = 400")
    plt.legend(loc="lower right")
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-3, 1e0 + 0.5])
    plt.xlabel('t')
    plt.ylabel('mse 2LNN')

    plt.subplot(3, 2, 6)
    plt.plot(t, mse_RF_200_mixed, 'red', label="D = 200")
    plt.plot(t, mse_RF_400_mixed, 'black', label="D = 400")
    plt.legend(loc="lower right")
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-3, 1e0 + 0.5])
    plt.xlabel('t')
    plt.ylabel('mse RF')
    plt.show()

    ######### Plot Input and Feature Space ###############################
    ######################################################################

    # Comment: As F is a random transformation matrix, the Feature Space will
    # not be the exact same one as shown ontop. Feel free to run this a couple
    # of times to get a figure as close as possible to the one displayed.

    plot_input_feature_spaces(dim=10, sigma=math.sqrt(0.05), mu_r_1=math.sqrt(10) * 2, mu_r_2=10 / 5)
    plot_input_feature_spaces(dim=10, sigma=math.sqrt(0.05), mu_r_1=math.sqrt(10) * 5, mu_r_2=math.sqrt(10) * 5)
    plot_input_feature_spaces(dim=10, sigma=math.sqrt(0.05), mu_r_1=math.sqrt(10) * 1, mu_r_2=math.sqrt(10) * 1)



