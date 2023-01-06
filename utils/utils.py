import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from utils.models import transform_RF

code_path = os.path.dirname(os.path.realpath('__file__'))
path = os.path.join(code_path, 'Data')


def make_GMM(dim, N, var, plot, mu_r_1=None, mu_r_2=None):
    '''
    This Function generates the gaussian mixtrue models. Set plot = True to inspect the first four dimensions visually.
    input:  dim     = dimension D
            N       = number of samples
            var     = standard deviation (sigma) for all clusters
            mu_r_1  = scaling facor for the distance of the cluster centers to the origin, default=None
            mu_r_2  = scaling facor for the distance of the cluster centers to the origin, default=None
    output: X       = data points of shape [N, dim]
            Y       = labels of shape [N]
            mus     = means of the 4 GMs of shape [4, dim]
    '''
    if mu_r_1 == None:
        mu_r_1 = math.sqrt(dim)
    if mu_r_2 == None:
        mu_r_2 = math.sqrt(dim)

    # Cluster means of the 4 GMs in the first two dimensions.
    # If mu_r is set to none, then the cluster centers will be (0,±1) and (±1, 0).

    mu1 = [0, mu_r_2 / math.sqrt(dim)]
    mu2 = [0, (-1) * mu_r_2 / math.sqrt(dim)]
    mu3 = [mu_r_1 / math.sqrt(dim), 0]
    mu4 = [(-1) * mu_r_1 / math.sqrt(dim), 0]

    # Cluster means of the 4 GMs for the other D - 2 dimensions set to zero.
    if dim > 2:
        mu1 = np.append(mu1, np.zeros((dim - 2), dtype=int))
        mu2 = np.append(mu2, np.zeros((dim - 2), dtype=int))
        mu3 = np.append(mu3, np.zeros((dim - 2), dtype=int))
        mu4 = np.append(mu4, np.zeros((dim - 2), dtype=int))

    # Shared diagonal coariance matrix.

    cov = np.eye(dim) * (var ** 2)

    # Sampled datapoints from the 4 multivariate gaussians.

    cluster1 = np.random.multivariate_normal(mu1, cov, size=int(N / 4), check_valid='warn', tol=1e-8)
    cluster2 = np.random.multivariate_normal(mu2, cov, size=int(N / 4), check_valid='warn', tol=1e-8)
    cluster3 = np.random.multivariate_normal(mu3, cov, size=int(N / 4), check_valid='warn', tol=1e-8)
    cluster4 = np.random.multivariate_normal(mu4, cov, size=int(N / 4), check_valid='warn', tol=1e-8)

    # Labels for the 4 GMs according to the 2 clusters of an XOR distribution.
    label1 = np.ones(int(N / 4), dtype=int) * (-1)
    label2 = np.ones(int(N / 4), dtype=int) * (-1)
    label3 = np.ones(int(N / 4), dtype=int) * (1)
    label4 = np.ones(int(N / 4), dtype=int) * (1)

    if plot == True:
        # This part visualizes the first four dimensions of the data.

        plt.scatter(cluster1[:, 0], cluster1[:, 1], color='red')
        plt.scatter(cluster2[:, 0], cluster2[:, 1], color='red')
        plt.scatter(cluster3[:, 0], cluster3[:, 1], color='blue')
        plt.scatter(cluster4[:, 0], cluster4[:, 1], color='blue')
        plt.title('Input Space')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.gca().set_xticks([])
        plt.xticks([])
        plt.gca().set_yticks([])
        plt.yticks([])
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
        plt.show()

    return np.vstack((cluster1, cluster2, cluster3, cluster4)), np.hstack((label1, label2, label3, label4)), np.vstack(
        (mu1, mu2, mu3, mu4))


def make_splits(X, Y):
    '''
    input:  X       = data points of shape [N, dim]
            Y       = labels of shape      [N]
    output: X_train = 2/3 of the datapoints used for trainig of shape     [2N/3, dim]
            X_val   = 1/3 of the datapoints used for validation of shape  [N/3, dim]
            Y_train = 2/3 of the labels used for trainig of shape         [2N/3]
            Y_val   = 1/3 of the labels used for validation of shape      [N/3]
    '''
    N = np.shape(Y)[0]
    indices = np.arange(N)
    np.random.shuffle(indices)

    X = X[indices]
    Y = Y[indices]
    X_train = X[0:int(N * 0.66), :]
    X_val = X[int(N * 0.66):, :]
    Y_train = Y[0:int(N * 0.66)]
    Y_val = Y[int(N * 0.66):]

    return torch.from_numpy(X_train), torch.from_numpy(X_val), torch.from_numpy(Y_train), torch.from_numpy(Y_val)


def plot_input_feature_spaces(dim, sigma, mu_r_1, mu_r_2):
    N = 500
    X, Y, m = make_GMM(dim, N, var=sigma, plot=True, mu_r_1=mu_r_1, mu_r_2=mu_r_2)
    F = torch.randn((dim, dim * 10))
    X_trafo = transform_RF(X=X, F=F)
    fig = plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 16})
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_trafo[:int(N / 2), 0], X_trafo[:int(N / 2), 1], X_trafo[:int(N / 2), 2], color='red')
    ax.scatter(X_trafo[int(N / 2):int(N), 0], X_trafo[int(N / 2):int(N), 1], X_trafo[int(N / 2):int(N), 2],
               color='blue')
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    frame1.axes.zaxis.set_ticklabels([])
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_zlabel('z3')
    plt.title('Feature Space of RF')
    plt.show()