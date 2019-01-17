import numpy as np
import tensorflow as tf
import gpflow
from IPython.display import display
from pilco.models.mgpr import MGPR
from abc import ABC, abstractmethod
float_type = gpflow.settings.dtypes.float_type


class Policy(ABC):
    @abstractmethod
    def compute_action(self, m_x, s_x):
        return NotImplementedError


def squash_sin(m, s, max_action=None):
    '''
    Squashing function, passing the controls mean and variance
    through a sinus, as in gSin.m. The output is in [-max_action, max_action].
    IN: mean (m) and variance(s) of the control input, max_action
    OUT: mean (M) variance (S) and input-output (C) covariance of the squashed
         control input
    '''
    k = tf.shape(m)[1]
    if max_action is None:
        max_action = tf.ones((1, k),
                             dtype=float_type)  #squashes in [-1,1] by default
    else:
        max_action = max_action * tf.ones((1, k), dtype=float_type)

    M = max_action * tf.exp(-tf.diag_part(s) / 2) * tf.sin(m)

    lq = -(tf.diag_part(s)[:, None] + tf.diag_part(s)[None, :]) / 2
    q = tf.exp(lq)
    S = (tf.exp(lq + s) - q) * tf.cos(tf.transpose(m) - m) \
        - (tf.exp(lq - s) - q) * tf.cos(tf.transpose(m) + m)
    S = max_action * tf.transpose(max_action) * S / 2

    C = max_action * tf.diag(tf.exp(-tf.diag_part(s) / 2) * tf.cos(m))
    return M, S, tf.reshape(C, shape=[k, k])


class LinearPolicy(Policy):
    """
    Linear Preliminary Policy
    See Deisenroth et al 2015: Gaussian Processes for Data-Efficient Learning in Robotics and Control
    Section 3.5.2 (pg 43)
    """

    def __init__(self, state_dim, control_dim, max_action):
        self.max_action = max_action
        self.Phi = np.random.rand(
            control_dim, state_dim)  # parameter matrix of weights (n, D)
        self.v = np.random.rand(1, control_dim)  # offset/bias vector (1, D )

    def compute_action(self, m, s):
        '''
        Predict Gaussian distribution for action given a state distribution input
        :param m: mean of the state
        :param s: variance of the state
        :return: mean (M) and variance (S) of action
        '''
        M = np.dot(self.Phi, m) + self.v
        S = np.dot(self.Phi, s).dot(self.Phi.T)
        return M, S


class PseudGPR(gpflow.Parameterized):
    def __init__(self, X, Y, kernel, name="PseudoGPR"):
        gpflow.Parameterized.__init__(self)
        self.X = gpflow.Param(X)
        self.Y = gpflow.Param(Y)
        self.kern = kernel
        self.likelihood = gpflow.likelihoods.Gaussian()


class RBFNPolicy(MGPR, Policy):
    """
    Radial Basis Function Network Preliminary Policy
    See Deisenroth et al 2015: Gaussian Processes for Data-Efficient Learning in Robotics and Control
    Section 3.5.2 (pg 44)
    """

    def __init__(self,
                 state_dim,
                 control_dim,
                 num_basis_fun,
                 max_action,
                 name="RBFN Policy"):
        """
        Radial Basis Function Network
        :param state_dim: shape of input data (state_dim + control_dim)
        :param num_basis_fun: number of radial basis functions (no. hidden neurons)
        :param sigma:
        """
        MGPR.__init__(self, 0.1 * np.random.randn(num_basis_fun, state_dim),
                      0.01 * np.random.randn(num_basis_fun, control_dim))

        for model in self.models:
            model.kern.variance = 1.0
            model.kern.variance.trainable = False
            self.max_action = max_action

    def create_models(self, x_train, y_train):
        self.models = gpflow.params.ParamList([])
        for i in range(y_train.shape[1]):
            kernel = gpflow.kernels.RBF(input_dim=x_train.shape[1], ARD=True)
            model = PseudGPR(x_train, y_train[:, i:i + 1], kernel)
            self.models.append(model)

    def compute_action(self, m, s, squash=True):
        '''
        RBF Controller. See Deisenroth's Thesis Section
        IN: mean (m) and variance (s) of the state
        OUT: mean (M) and variance (S) of the action
        '''
        iK, beta = self.calculate_factorizations()
        M, S, V = self.predict_given_factorizations(m, s, 0.0 * iK, beta)
        S = S - tf.diag(self.variance - 1e-6)
        if squash:
            M, S, V2 = squash_sin(M, S, self.max_action)
            V = V @ V2
        return M, S, V

    # def compute_action(self, m_x, s_x):
    #     iK, beta = self.calculate_factorizations()
    #     m_u, s_u, v = self.predict_given_factorizations(
    #         m_x, s_x, 0.0 * iK, beta)
    #     # m_u[m_u > 1] = 1
    #     # m_u[m_u < -1] = -1
    #     return m_u, s_u, v
