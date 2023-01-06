import torch
import sklearn.cluster
import torch.nn as nn
from torch import optim
from torch import relu as relu
import math
import numpy as np
def HalfMSE(output, target):
    loss = (0.5)*torch.mean((output - target)**2)
    return loss

# Oracle
def oracle(X, mu):
    """
    This function implements the 'oracle' which is defined as a network "with knowledge of the means of
    the mixture that assigns to each input the label of the nearest mean".

    Input:  X       = data points of shape [N, dim]
            mu      = means of the 4 GMs of shape [4, dim]
    Output: labels  = assigned cluster to each datapoints of shape [N]
    """
    oracle = sklearn.cluster.KMeans(n_clusters=4, init=mu, n_init=1).fit(X)
    labels = oracle.labels_
    ind1 = np.where(labels == 0)[0]
    ind2 = np.where(labels == 1)[0]
    ind3 = np.where(labels == 2)[0]
    ind4 = np.where(labels == 3)[0]

    cluster1 = np.hstack((ind1, ind2))
    cluster2 = np.hstack((ind3, ind4))

    labels[labels == 0] = -1
    labels[labels == 1] = -1
    labels[labels == 2] = 1
    labels[labels == 3] = 1

    return labels

# 2 layer NN
class Student(nn.Module):
    """
    This is the 2-layerd neuronal network with K hidden neurons and 1 output neuron, used thoughtout this report.
    """

    def __init__(self, K, N, weight_std_initial_layer=1):
        """
        Input:  K                         = number of hidden neurons
                N                         = number of samples
                weight_std_initial_layer  = standard deviation for the weight initialization of the first
        """
        print("Creating a Student with InputDimension: %d, K: %d" % (N, K))
        super(Student, self).__init__()

        self.N = N
        self.g = nn.ReLU()
        self.K = K
        self.loss = nn.MSELoss(reduction='mean')
        # Definition of the 2 layers
        self.fc1 = nn.Linear(N, K, bias=False)
        self.fc2 = nn.Linear(K, 1, bias=False)

        ##For Figure 1 reproduction
        # torch.nn.init.xavier_uniform_(self.fc2.weight)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.weight)
        nn.init.normal_(self.fc2.weight)

        ##For figure 4 reproduction
        # nn.init.normal_(self.fc1.weight,std=weight_std_initial_layer)
        # nn.init.normal_(self.fc2.weight,std=weight_std_initial_layer)

    def forward(self, x):
        # This is the input to the hidden layer.
        x = self.fc1(x) / math.sqrt(self.N)
        x = self.g(x)
        x = self.fc2(x)
        return x


# Random Features
def linear(x):
    return x


def centered_relu(x, var):
    a = math.sqrt(var) / math.sqrt(2 * math.pi)
    return torch.relu(x) - a


def transform_RF(X, F):
    """
    This function tansforms the datapoints X into a feature space of P>>dim, with the
    transform-matrix F.
    Input:  X       = data points of shape [N, dim]
            F       = transformation matrix of shape [dim, P]
    Output: X_trafo = transformed datapoints in the feature space of shape [N, P]
    """
    D, P = F.shape
    X = torch.from_numpy(X)
    X = X.float()
    F /= F.norm(dim=0).repeat(D, 1)
    F *= math.sqrt(D)
    X_trafo = centered_relu((X @ F) / math.sqrt(D), 0)
    return X_trafo


class Student_RF(nn.Module):
    """
    This is the second layer for the Random Features, which takes the projected datapoints
    and predcits the cluster labels via a linear model.
    """

    def __init__(self, K, N, bias=False):
        """
        Input:  K                         = number of hidden neurons
                N                         = number of samples
        """
        print("Creating a Student with InputDimension: %d, K: %d" % (N, K))
        super(Student_RF, self).__init__()

        self.P = N
        self.g = linear
        self.K = 1
        self.loss = nn.MSELoss(reduction='mean')
        self.fc1 = nn.Linear(self.P, K, bias)
        nn.init.normal_(self.fc1.weight, std=0.01)

    def forward(self, x):
        x = self.g(self.fc1(x) / math.sqrt(self.P))
        return x