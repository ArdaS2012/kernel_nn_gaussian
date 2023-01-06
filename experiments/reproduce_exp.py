import math

import sklearn
import torch
from torch import optim

from plots.plot_figures import plot_figure1, plot_figure4, plot_figure5, plot_figure6
from utils.models import oracle, HalfMSE, transform_RF, Student_RF, Student
from utils.utils import make_GMM, make_splits
import numpy as np


def exp_1(N,dim_NN,dim_RF,dim_oracle):
    def log_sigmas(num_sigmas):
        """
        Defines the sigmas, that will be used to generate Figure 1.
        """
        sigma1 = np.logspace(-2, -1, num=int(num_sigmas / 3))
        sigma2 = np.logspace(-1, 0, num=int(num_sigmas / 3))
        sigma3 = np.logspace(0, 1, num=int(num_sigmas / 3))
        sigma = np.round(np.append(sigma1, np.append(sigma2, sigma3)), 5)
        return sigma


    ######### definition of input parameters #############################
    ######################################################################

    num_sigmas = 15 * 3
    sigma = log_sigmas(num_sigmas)

    """
    Oracle: 
    Here, we compute the predictions and classification error for the oracle. The dimension
    is fixed but we iterate though different standard deviations for the gaussians. The gaussians
    are i.i.d. meaning, the cluster centers have a 1 distance to the origin and are not scaled. 
    """

    oracle_pred = np.zeros((num_sigmas, int(N)))
    oracle_error = np.zeros((num_sigmas))

    for i in range(0, num_sigmas):
        X, Y, mu = make_GMM(dim=dim_oracle, N=N, var=sigma[i], plot=True)
        labels = oracle(X, mu)
        ind1 = np.where(labels == 1)[0]
    ind2 = np.where(labels == -1)[0]
    oracle_error[i] = sklearn.metrics.zero_one_loss(Y, labels, normalize=True, sample_weight=None)
    oracle_pred[i, :] = labels
    # print('Classifiaction error: {} for sigma: {}'.format(np.round(oracle_error[i], 3),np.round(sigma[i], 3)))

    """
    Random Features: 
    Here, we compute the classification error for the Random Features. The dimension
    is fixed but we iterate though different standard deviations for the gaussians. The gaussians
    are i.i.d. meaning, the cluster centers have a 1 distance to the origin and are not scaled. 
    """

    P = 2 * dim_RF  # projection dimension
    reg_RF = 0.0  # regulaization parameter
    lr = 0.5  # learning rate
    RF_error = np.zeros((num_sigmas))

    ######### initilize the second layer  for RF #########################
    ######################################################################
    student = Student_RF(N=P, K=1)
    params = []
    params += [{'params': student.fc1.parameters(), 'lr': lr, 'weight_decay': reg_RF}]
    optimizer = optim.SGD(params, lr=lr, weight_decay=reg_RF)
    criterion = student.loss

    ######### iterate over the sigmas  ###################################
    ######################################################################

    for i in range(0, num_sigmas):
        X_, Y_, mu = make_GMM(dim=dim_RF, N=N, var=sigma[i], plot=False)
        F = torch.randn((dim_RF, P))  # random, but fixed projection matrix
        X = transform_RF(X_, F)  # projected data
        X = (X).numpy()

        X_train, X_val, Y_train, Y_val = make_splits(X, Y)
        X_val = (X_val).float()
        Y_val = (Y_val).float()

        ######### Training the RF with online SGD on halfMSE #################
        ######################################################################
        student.train()
        for j in range(X_train.shape[0]):
            targets = (Y_train[j]).float()
            inputs = (X_train[j, :]).float()
            student.zero_grad()
            preds = student(inputs)
            loss = HalfMSE(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
            optimizer.step()

        ######### Evaluation of the training of RF on the classification error
        ######################################################################
        student.eval()
        with torch.no_grad():
            preds = student(X_val)
            preds = preds[:, 0]
            eg = HalfMSE(preds, Y_val)
            # calculate the classification error with the predictions
            eg_class = 1 - torch.relu(torch.sign(preds) * Y_val)
            eg_class = eg_class.sum() / float(preds.shape[0])
            # print("preds:{}, y_val:{}".format(preds,Y_val))
            RF_error[i] = eg_class
            print("Test Data: Classification Error: {}; Variance: {}; halfMSE-Loss:{}".format(np.round(RF_error[i], 3),
                                                                                              np.round(sigma[i], 3),
                                                                                              eg))
            print("---------------------------------------------------------")

    """
    2 Layer NN: 
    Here, we compute the classification error for the 2LNN. The dimension
    is fixed but we iterate though different standard deviations for the gaussians. The gaussians
    are i.i.d. meaning, the cluster centers have a 1 distance to the origin and are not scaled. 
    """

    K = 12  # number of hidden neurons
    lr_NN = 0.1  # learning rate
    reg_NN = 0.0  # regulaization parameter
    NN_error = np.zeros((num_sigmas))

    ######### initilize the 2LNN #########################################
    ######################################################################
    student = Student(K=K, N=dim_NN)
    params = []
    params += [{'params': student.fc1.parameters()}]
    params += [{'params': student.fc2.parameters(), 'lr': lr_NN, 'weight_decay': reg_NN}]
    optimizer = optim.SGD(params, lr=lr_NN,
                          weight_decay=reg_NN)  # Define which parameters should be optimized by the SGD
    criterion = student.loss

    ######### iterate over the sigmas  ###################################
    ######################################################################
    for i in range(0, num_sigmas):
        X, Y, m = make_GMM(dim=dim_NN, N=N, var=sigma[i], plot=False)  # Generate new data with new variance
        X_train, X_val, Y_train, Y_val = make_splits(X, Y)
        X_train = X_train
        Y_train = Y_train
        X_val = (X_val).float()
        Y_val = (Y_val).float()

        ######### Training the NN with online SGD on halfMSE #################
        ######################################################################
        student.train()
        for j in range(X_train.shape[0]):
            targets = (Y_train[j]).float()
            inputs = (X_train[j, :]).float()
            student.zero_grad()
            preds = student(inputs)
            loss = HalfMSE(preds, targets)
            # if j% 500 ==0: #print train loss every 100 steps
            #  print("Train loss: {}".format(loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()

        ######### Evaluation of the training of NN on the classification error
        ######################################################################
        student.eval()
        with torch.no_grad():
            preds = student(X_val)
            preds = preds[:, 0]
            eg = HalfMSE(preds, Y_val)
            # calculate the classification error with the predictions
            eg_class = 1 - torch.relu(torch.sign(preds) * Y_val)
            eg_class = eg_class.sum() / float(preds.shape[0])
            NN_error[i] = eg_class
            print("Test Data: Generalized Classification Error: {}; Variance: {}; Loss:{}".format(
                np.round(NN_error[i], 3), np.round(sigma[i], 3), eg))
            print("---------------------------------------------------------")

    plot_figure1(oracle_error,RF_error,NN_error,sigma,dim_RF)

def exp_4(K_arr,dim,N,lr_NN,reg_NN,nr_sim_total):
    ######### define the parameters and create the dataset ###############
    ######################################################################
    var = math.sqrt(0.1)
    X, Y, mu = make_GMM(dim=dim, N=N, var=var, plot=False)  # Generate new data with new variance
    X_train, X_val, Y_train, Y_val = make_splits(X, Y)
    labels = oracle(X, mu)
    oracle_error_fig4 = sklearn.metrics.zero_one_loss(Y, labels, normalize=True, sample_weight=None)
    def run_model(std_weights,X_val,Y_val):
        """
        Performance Analysis for different K and initial std of weights == 0.01 and 1.0
        :param std_weights: initialized std for weights
        :return: fraction of converged runs to total simulation nr
        """
        NN_error_fig41 = np.zeros((len(K_arr), nr_sim_total))
        X_val = (X_val).float()
        Y_val = (Y_val).float()
        result_001 = np.zeros((len(K_arr)))
        ######### iterate over different no. neurons ##########################
        ######################################################################
        for i in range(0, len(K_arr)):
            K = K_arr[i]
            nr_sim_converged = 0
            for j in range(nr_sim_total):
                student = Student(K=K, N=dim,weight_std_initial_layer=std_weights)
                params = []
                params += [{'params': student.fc1.parameters()}]
                params += [{'params': student.fc2.parameters(), 'lr': lr_NN, 'weight_decay': reg_NN}]
                optimizer = optim.SGD(params, lr=lr_NN, weight_decay=reg_NN)

                ######### Training the NN with online SGD on halfMSE #################
                ######################################################################
                student.train()
                for z in range(X_train.shape[0]):
                    targets = (Y_train[z]).float()
                    inputs = (X_train[z, :]).float()
                    student.zero_grad()
                    preds = student(inputs)
                    loss = HalfMSE(preds, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
                    optimizer.step()

                ######### Evaluation of the training of NN on the classification error
                ######################################################################
                student.eval()
                with torch.no_grad():
                    preds = student(X_val)
                    preds = preds[:, 0]
                    eg = HalfMSE(preds, Y_val)
                    # calculate the classification error with the predictions
                    eg_class = 1 - torch.relu(torch.sign(preds) * Y_val)
                    eg_class = eg_class.sum() / float(preds.shape[0])
                    NN_error_fig41[i, j] = eg_class
                    if np.abs(NN_error_fig41[i, j] - oracle_error_fig4) <= 0.05:
                        nr_sim_converged = nr_sim_converged + 1
                    print("Class. Error: {} for simulation: {} and K:{}".format(NN_error_fig41[i, j], j, K))
            result_001[i] = np.round(nr_sim_converged / nr_sim_total, 3)
            print("{} out of {} simulations converged:{} ".format(nr_sim_converged, nr_sim_total, result_001[i]))
            print("---------------------------------------------------------")
        return result_001

    ######### initilize the 2LNN #########################################
    ######################################################################
    result_001 = run_model(0.01)
    result_100 = run_model(1.0)
    #### plot result for experiment 4
    plot_figure4(result_100,result_001,K_arr)


def exp_5(N,dim_RF,gamma,P,sigma,reg_RF,lr,num_sigmas):
    def performane_over_gamma(gamma, dim, P,num_sigmas):
        ######### initilize the NN layers for RF #############################
        ######################################################################
        print("Input Dimension: {}, Nr of Inputs: {}, P:{}, gamma:{}".format(dim, N, P, gamma))
        student = Student_RF(N=P, K=1)
        params = []
        params += [{'params': student.fc1.parameters(), 'lr': lr, 'weight_decay': reg_RF}]
        optimizer = optim.SGD(params, lr=lr, weight_decay=reg_RF)
        criterion = student.loss
        error_fig5_RF = np.zeros((num_sigmas))

        ######### iterate over sigmas ########################################
        ######################################################################
        for i in range(0, num_sigmas):
            print("{} out of {} runs".format(i + 1, num_sigmas))
            X, Y, m = make_GMM(dim=dim, N=N, var=sigma[i], plot=False)
            F = torch.randn((dim, P))
            X = transform_RF(X=X, F=F)
            X = (X).numpy()
            X_train, X_val, Y_train, Y_val = make_splits(X, Y)
            X_val = (X_val).float()
            Y_val = (Y_val).float()

            ######### Training the RF with online SGD on HalfMSE #################
            ######################################################################
            student.train()
            for j in range(X_train.shape[0]):
                targets = (Y_train[j]).float()
                inputs = (X_train[j, :]).float()
                student.zero_grad()
                preds = student(inputs)
                loss = HalfMSE(preds, targets)
                if j % 500 == 0:
                    print("Train loss: {}".format(loss))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
                optimizer.step()
            ######### Evaluation the RF on classification error ##################
            ######################################################################
            student.eval()
            with torch.no_grad():
                preds = student(X_val)
                preds = preds[:, 0]
                eg = HalfMSE(preds, Y_val)
                eg_class = 1 - torch.relu(torch.sign(preds) * Y_val)
                eg_class = eg_class.sum() / float(preds.shape[0])
                print("preds:{}, y_val:{}".format(preds, Y_val))
                error_fig5_RF[i] = eg_class
                print("Test: Classification Error: {}; Variance: {}; Loss:{}".format(np.round(error_fig5_RF[i], 3),
                                                                                     sigma[i], eg))
                print("---------------------------------------------------------")

        print("Training sucessfully finished!")
        return error_fig5_RF


    result_gamma_8_dim_400 = performane_over_gamma(gamma[0], dim_RF[0], P[0],num_sigmas)
    result_gamma_5_dim_300 = performane_over_gamma(gamma[1], dim_RF[1], P[1],num_sigmas)
    result_gamma_2_dim_1000 = performane_over_gamma(gamma[2], dim_RF[2], P[2],num_sigmas)
    plot_figure5(num_sigmas,result_gamma_2_dim_1000,result_gamma_5_dim_300,result_gamma_8_dim_400)


def exp_6(t,N1,N2,P1,P2,K,dim1,dim2,lr,reg,sigma):
    def iterate_over_time(N, P, K, dim, lr, reg, sigma, mu_r_1, mu_r_2):

        ######### initilize the NN layers for RF and 2LNN ####################
        ######################################################################
        student = Student_RF(N=P, K=1)
        student_2lnn = Student(K=K, N=dim)
        params_RF = []
        params_RF += [{'params': student.fc1.parameters(), 'lr': lr, 'weight_decay': reg}]
        optimizer_RF = optim.SGD(params_RF, lr=lr, weight_decay=reg)
        criterion_RF = student.loss
        params_2LNN = []
        params_2LNN += [{'params': student_2lnn.fc1.parameters()}]
        params_2LNN += [{'params': student_2lnn.fc2.parameters(), 'lr': lr, 'weight_decay': reg}]
        optimizer_2LNN = optim.SGD(params_2LNN, lr=lr, weight_decay=reg)
        criterion_2LNN = student_2lnn.loss

        mse_RF = np.zeros((len(N)))
        mse_2LNN = np.zeros((len(N)))
        print("For length N = {}".format(len(N)))

        ######### iterate over time (t = N/D) ################################
        ######################################################################
        for i in range(0, len(N)):
            print("Run number:{}".format(i))
            X, Y, m = make_GMM(dim=dim, N=int(N[i]), var=sigma, plot=False, mu_r_1=mu_r_1, mu_r_2=mu_r_2)
            F = torch.randn((dim, P))
            X_RF = transform_RF(X, F)
            X_RF = (X_RF).numpy()
            X_train_RF, X_val_RF, Y_train_RF, Y_val_RF = make_splits(X_RF, Y)
            X_train_2LNN, X_val_2LNN, Y_train_2LNN, Y_val_2LNN = make_splits(X, Y)
            X_val_RF = (X_val_RF).float()
            Y_val_RF = (Y_val_RF).float()
            X_val_2LNN = (X_val_2LNN).float()
            Y_val_2LNN = (Y_val_2LNN).float()

            ######### Training the 2LNN and RF with online SGD on MSE ############
            ######################################################################
            student.train()
            student_2lnn.train()
            for j in range(X_train_RF.shape[0]):
                targets_RF = (Y_train_RF[j]).float()
                inputs_RF = (X_train_RF[j, :]).float()
                targets_2LNN = (Y_train_2LNN[j]).float()
                inputs_2LNN = (X_train_2LNN[j, :]).float()
                student.zero_grad()
                student_2lnn.zero_grad()
                preds_RF = student(inputs_RF)
                preds_2LNN = student_2lnn(inputs_2LNN)
                loss_RF = criterion_RF(preds_RF, targets_RF)
                loss_2LNN = criterion_2LNN(preds_2LNN, targets_2LNN)
                if j % 500 == 0:  # print train loss every 100 steps
                    print("Train loss RF: {}--- Train loss 2LNN:{}".format(loss_RF, loss_2LNN))
                loss_RF.backward()
                loss_2LNN.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
                torch.nn.utils.clip_grad_norm_(student_2lnn.parameters(), 10.0)
                optimizer_RF.step()
                optimizer_2LNN.step()

            ######### Evaluating the 2LNN and RF with online SGD on MSE ###########
            ######################################################################
            student.eval()
            student_2lnn.eval()
            with torch.no_grad():
                preds_RF = student(X_val_RF)
                preds_2LNN = student_2lnn(X_val_2LNN)
                preds_RF = preds_RF[:, 0]
                preds_2LNN = preds_2LNN[:, 0]
                eg_RF = criterion_RF(preds_RF, Y_val_RF)
                eg_2LNN = criterion_2LNN(preds_2LNN, Y_val_2LNN)
                mse_RF[i] = eg_RF
                mse_2LNN[i] = eg_2LNN
                _eg_RF = eg_RF.cpu().detach().numpy()
                _eg_2LNN = eg_2LNN.cpu().detach().numpy()
                t_ = N[i] / dim
                print("Test Data: MSE RF: {}; MSE 2LNN:{} t:{}".format(np.round(_eg_RF, 3), np.round(_eg_2LNN, 3),
                                                                       int(t_)))
                print("---------------------------------------------------------")
        return mse_RF, mse_2LNN

    ############# low SNR ################################################
    mse_RF_200_low, mse_2LNN_200_low = iterate_over_time(N1, P1, K, dim1, lr, reg, sigma, mu_r_1=math.sqrt(dim1) * 1,
                                                         mu_r_2=math.sqrt(dim1) * 1)
    mse_RF_400_low, mse_2LNN_400_low = iterate_over_time(N2, P2, K, dim2, lr, reg, sigma, mu_r_1=math.sqrt(dim2) * 1,
                                                         mu_r_2=math.sqrt(dim2) * 1)
    ############# high SNR ###############################################
    mse_RF_200_high, mse_2LNN_200_high = iterate_over_time(N1, P1, K, dim1, lr, reg, sigma, mu_r_1=math.sqrt(dim1) * 5,
                                                           mu_r_2=math.sqrt(dim1) * 5)
    mse_RF_400_high, mse_2LNN_400_high = iterate_over_time(N2, P2, K, dim2, lr, reg, sigma, mu_r_1=math.sqrt(dim2) * 5,
                                                           mu_r_2=math.sqrt(dim2) * 5)
    ############# mixed SNR ##############################################
    mse_RF_200_mixed, mse_2LNN_200_mixed = iterate_over_time(N1, P1, K, dim1, lr, reg, sigma,
                                                             mu_r_1=math.sqrt(dim1) * 2, mu_r_2=dim1 / 20)
    mse_RF_400_mixed, mse_2LNN_400_mixed = iterate_over_time(N2, P2, K, dim2, lr, reg, sigma,
                                                             mu_r_1=math.sqrt(dim2) * 2, mu_r_2=dim2 / 20)

    plot_figure6(t,dim1,dim2,mse_2LNN_200_low,mse_2LNN_400_low,mse_RF_200_low,mse_RF_400_low,mse_2LNN_200_high,mse_2LNN_400_high,
                 mse_RF_200_high, mse_RF_400_high, mse_2LNN_200_mixed, mse_2LNN_400_mixed, mse_RF_200_mixed, mse_RF_400_mixed,
                 )