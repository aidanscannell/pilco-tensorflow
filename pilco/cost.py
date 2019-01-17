import abc
import tensorflow as tf
from gpflow import Parameterized, Param, params_as_tensors, settings
import numpy as np

float_type = settings.dtypes.float_type


class Cost(Parameterized):
    def __init__(self):
        Parameterized.__init__(self)

    @abc.abstractmethod
    def compute_cost(self, m, s):
        raise NotImplementedError


class SaturatingCost(Cost):
    def __init__(self, state_dim, state_idxs=None, W=None, t=None):
        Cost.__init__(self)
        self.state_dim = state_dim
        if W is not None:
            self.W = Param(
                np.reshape(W, (state_dim, state_dim)), trainable=False)
        else:
            self.W = Param(np.ones((state_dim, state_dim)), trainable=False)
        if t is not None:
            self.t = Param(np.reshape(t, (1, state_dim)), trainable=False)
        else:
            self.t = Param(np.zeros((1, state_dim)), trainable=False)

        if state_idxs is None:
            state_idxs = np.linspace(
                0, state_dim - 1, num=state_dim, dtype=int)

        self.m_idxs = []
        self.s_idxs = []

        for index in state_idxs:
            self.m_idxs.append([0, index])
            element = []
            for j in state_idxs:
                element.append([index, j])
            self.s_idxs.append(element)

    @params_as_tensors
    def compute_cost(self, m_x, s_x):
        '''
        Reward function, calculating mean and variance of rewards, given
        mean and variance of state distribution, along with the target State
        and a weight matrix.
        Input m : [1, k]
        Input s : [k, k]

        Output M : [1, 1]
        Output S  : [1, 1]
        '''
        # TODO: Clean up this

        m = tf.gather_nd(m_x, self.m_idxs)
        s = tf.gather_nd(s_x, self.s_idxs)

        SW = s @ self.W

        iSpW = tf.transpose(
            tf.matrix_solve((tf.eye(self.state_dim, dtype=float_type) + SW),
                            tf.transpose(self.W),
                            adjoint=True))

        muR = tf.exp(-(m-self.t) @  iSpW @ tf.transpose(m-self.t)/2) / \
                tf.sqrt( tf.linalg.det(tf.eye(self.state_dim, dtype=float_type) + SW) )

        i2SpW = tf.transpose(
            tf.matrix_solve(
                (tf.eye(self.state_dim, dtype=float_type) + 2 * SW),
                tf.transpose(self.W),
                adjoint=True))

        r2 =  tf.exp(-(m-self.t) @ i2SpW @ tf.transpose(m-self.t)) / \
                tf.sqrt( tf.linalg.det(tf.eye(self.state_dim, dtype=float_type) + 2*SW) )

        sR = r2 - muR @ muR
        muR.set_shape([1, 1])
        sR.set_shape([1, 1])
        return muR, sR


# from abc import abstractmethod
# import tensorflow as tf
# from gpflow import Parameterized, Param, params_as_tensors, settings
# import numpy as np
#
# float_type = settings.dtypes.float_type
#
#
# class Cost(Parameterized):
#     def __init__(self):
#         Parameterized.__init__(self)
#
#     @abstractmethod
#     def compute_cost(self, m, s):
#         raise NotImplementedError
#
#
# class SaturatingCost(Cost):
#     """Saturating cost as defined in
#     See Deisenroth et al 2015: Gaussian Processes for Data-Efficient Learning in Robotics and Control
#     Section 3.5.2 (pg 43)
#     """
#
#     # def __init__(self, state_dim, T=None, x_target=None):
#     #     Cost.__init__(self)
#     #     self.state_dim = state_dim
#     #     if T is not None:
#     #         self.T = Param(
#     #             np.reshape(T, (state_dim, state_dim)), trainable=False)
#     #     else:
#     #         self.T = Param(np.ones((state_dim, state_dim)), trainable=False)
#     #     if x_target is not None:
#     #         self.x_target = Param(
#     #             np.reshape(x_target, (1, state_dim)), trainable=False)
#     #     else:
#     #         self.x_target = Param(np.zeros((1, state_dim)), trainable=False)
#
#     def __init__(self, x_target, iT, state_dim, state_idxs=None):
#         Cost.__init__(self)
#         if iT is not None:
#             self.iT = Param(
#                 np.reshape(iT, (state_dim, state_dim)), trainable=False)
#         else:
#             self.iT = Param(np.ones((state_dim, state_dim)), trainable=False)
#         if x_target is not None:
#             self.x_target = Param(
#                 np.reshape(x_target, (1, state_dim)), trainable=False)
#         else:
#             self.x_target = Param(np.zeros((1, state_dim)), trainable=False)
#
#         self.state_dim = state_dim
#         self.m_idxs = []
#         self.s_idxs = []
#         self.idxs = state_idxs
#         for index in state_idxs:
#             self.m_idxs.append([0, index])
#             element = []
#             for j in state_idxs:
#                 element.append([index, j])
#             self.s_idxs.append(element)
#
#     @params_as_tensors
#     def compute_cost(self, m_x, s_x):
#         m_j = tf.gather_nd(m_x, self.m_idxs)
#         s_j = tf.gather_nd(s_x, self.s_idxs)
#         # return self.expected_immediate_cost(m_j, s_j)
#         return self.immediate_cost(m_j, s_j)
#
#     def immediate_cost(self, m_j, s_j):
#         return 1 - tf.exp(-0.5 * (m_j - self.x_target) @ self.iT
#                           @ tf.transpose(m_j - self.x_target))
#
#     def first_moment(self, m_j, s_j):
#         det = tf.eye(self.state_dim, dtype=float_type) + s_j @ self.iT
#         S1 = tf.transpose(
#             tf.matrix_solve(det, tf.transpose(self.iT), adjoint=True))
#         diff = tf.reshape(m_j - self.x_target, [1, 3])
#         E = 1 - tf.exp(-(diff @ S1 @ tf.transpose(diff)) / 2) / tf.sqrt(
#             tf.linalg.det(det))
#         return E
#
#     def second_moment(self, m_j, s_j):
#         det = tf.eye(self.state_dim, dtype=float_type) + 2 * s_j @ self.iT
#         S2 = tf.transpose(
#             tf.matrix_solve(det, tf.transpose(self.iT), adjoint=True))
#         diff = tf.reshape(m_j - self.x_target, [1, 3])
#         E2 = tf.exp(-(diff @ S2 @ tf.transpose(diff))) / tf.sqrt(
#             tf.linalg.det(det))
#         return E2
#
#     def expected_immediate_cost(self, m_j, s_j):
#         E = self.first_moment(m_j, s_j)
#         E2 = self.second_moment(m_j, s_j)
#         s_c = E2 - E @ E
#         E.set_shape([1, 1])
#         s_c.set_shape([1, 1])
#         return E, s_c
#         # return E.set_shape([1, 1]), s_c.set_shape([1, 1])
#
#     # @params_as_tensors
#     # def compute_reward(self, m, s):
#     #     '''
#     #     Reward function, calculating mean and variance of rewards, given
#     #     mean and variance of state distribution, along with the target State
#     #     and a weight matrix.
#     #     Input m : [1, k]
#     #     Input s : [k, k]
#     #
#     #     Output M : [1, 1]
#     #     Output S  : [1, 1]
#     #     '''
#     #     # TODO: Clean up this
#     #
#     #     ST = s @ self.T
#     #
#     #     det = tf.eye(self.state_dim, dtype=float_type) + ST
#     #
#     #     S1 = tf.transpose(
#     #         tf.matrix_solve(det, tf.transpose(self.T), adjoint=True))
#     #
#     #     E = tf.exp(-0.5 * (m - self.x_target) @ S1
#     #                @ tf.transpose(m - self.x_target)) / tf.sqrt(
#     #                    tf.linalg.det(det))
#     #
#     #     E2 = tf.transpose(
#     #         tf.matrix_solve(
#     #             (tf.eye(self.state_dim, dtype=float_type) + 2 * SW),
#     #             tf.transpose(self.W),
#     #             adjoint=True))
#     #
#     #     r2 =  tf.exp(-(m-self.x_target) @ i2SpW @ tf.transpose(m-self.x_target)) / \
#     #             tf.sqrt( tf.linalg.det(tf.eye(self.state_dim, dtype=float_type) + 2*SW) )
#     #
#     #     sR = r2 - muR @ muR
#     #     muR.set_shape([1, 1])
#     #     sR.set_shape([1, 1])
#     #     return muR, sR
