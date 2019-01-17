import numpy as np
import tensorflow as tf
import gpflow
float_type = gpflow.settings.dtypes.float_type


class MGPR(gpflow.Parameterized):
    """Multiple Gaussian Process regression with prediction on noisy inputs.
    
    Creates a separate GP for every output dimension.
    """

    def __init__(self, x_train, y_train, name=None):
        """
        Args:
            x_train: training inputs (num_data_points x num_inputs)
            y_train: training outputs/targets (num_data_points x num_outputs
            name: name of the model
        """
        super(MGPR, self).__init__(name)
        self.num_outputs = y_train.shape[1]
        self.num_inputs = x_train.shape[1]
        self.num_data_points = x_train.shape[0]

        self.models = []
        self.create_models(x_train, y_train)

    def create_models(self, x_train, y_train):
        for i in range(self.num_outputs):
            kernel = gpflow.kernels.RBF(input_dim=x_train.shape[1], ARD=True)
            model = gpflow.models.GPR(
                X=x_train, Y=y_train[:, i:i + 1], kern=kernel)
            # model.clear()
            model.compile()
            self.models.append(model)
            # display(model)

    def set_XY(self, X, Y):
        for i in range(len(self.models)):
            print("X shape before update:")
            print(self.models[i].X.shape)
            self.models[i].X = X
            self.models[i].Y = Y[:, i:i + 1]

    def optimize(self):
        opt = gpflow.train.ScipyOptimizer(options={'maxfun': 500})
        for model in self.models:
            opt.minimize(model)
            # display(model)

    def predict_on_noisy_inputs(self, m, s):
        iK, beta = self.calculate_factorizations()
        return self.predict_given_factorizations(m, s, iK, beta)

    def calculate_factorizations(self):
        K = self.K(self.X)
        batched_eye = tf.eye(
            tf.shape(self.X)[0],
            batch_shape=[self.num_outputs],
            dtype=float_type)
        L = tf.cholesky(K + self.noise[:, None, None] * batched_eye)
        iK = tf.cholesky_solve(L, batched_eye)
        Y_ = tf.transpose(self.Y)[:, :, None]

        beta = tf.cholesky_solve(L, Y_)[:, :, 0]
        return iK, beta

    def predict_given_factorizations(self, m, s, iK, beta):
        """
        
        Args:
            m: mean of input (1 x num_inputs)
            s: variance of inout (num_inputs x num_inputs)
            iK: (K + σ^{2} I)^{−1} (num_outputs x num_data_points x num_data_points)
            beta: (K + σ^{2} I)^{−1} y (num_data_points x num_data_points)

        Returns:
            mean (M) and variance (S) of the output and the input/output covariance (V)

        """
        s = tf.tile(s[None, None, :, :],
                    [self.num_outputs, self.num_outputs, 1, 1])
        inp = tf.tile(
            self.centralized_input(m)[None, :, :], [self.num_outputs, 1, 1])

        # Calculate M and V: mean and inv(s) times input-output covariance
        iL = tf.matrix_diag(1 / self.lengthscales)
        iN = inp @ iL
        B = iL @ s[0, ...] @ iL + tf.eye(self.num_inputs, dtype=float_type)

        # Redefine iN as in^T and t --> t^T
        # B is symmetric so its the same
        t = tf.linalg.transpose(
            tf.matrix_solve(B, tf.linalg.transpose(iN), adjoint=True), )

        lb = tf.exp(-tf.reduce_sum(iN * t, -1) / 2) * beta

        tiL = t @ iL

        c = self.variance / tf.sqrt(tf.linalg.det(B))

        M = (tf.reduce_sum(lb, -1) * c)[:, None]

        V = tf.matmul(tiL, lb[:, :, None], adjoint_a=True)[..., 0] * c[:, None]

        # Calculate S: Predictive Covariance
        R = s @ tf.matrix_diag(
            1 / tf.square(self.lengthscales[None, :, :]) +
            1 / tf.square(self.lengthscales[:, None, :])) + tf.eye(
                self.num_inputs, dtype=float_type)

        X = inp[None, :, :, :] / tf.square(self.lengthscales[:, None, None, :])

        X2 = -inp[:, None, :, :] / tf.square(
            self.lengthscales[None, :, None, :])

        Q = tf.matrix_solve(R, s) / 2

        Xs = tf.reduce_sum(X @ Q * X, -1)

        X2s = tf.reduce_sum(X2 @ Q * X2, -1)

        maha = -2 * tf.matmul(X @ Q, X2, adjoint_b=True) + \
               Xs[:, :, :, None] + X2s[:, :, None, :]

        k = tf.log(self.variance)[:, None] - \
            tf.reduce_sum(tf.square(iN), -1) / 2

        L = tf.exp(k[:, None, :, None] + k[None, :, None, :] + maha)

        S = (tf.tile(beta[:, None, None, :], [1, self.num_outputs, 1, 1]) @ L
             @ tf.tile(beta[None, :, :, None],
                       [self.num_outputs, 1, 1, 1]))[:, :, 0, 0]

        diagL = tf.transpose(tf.linalg.diag_part(tf.transpose(L)))
        S = S - tf.diag(tf.reduce_sum(tf.multiply(iK, diagL), [1, 2]))
        S = S / tf.sqrt(tf.linalg.det(R))
        S = S + tf.diag(self.variance)
        S = S - M @ tf.transpose(M)

        return tf.transpose(M), S, tf.transpose(V)

    def centralized_input(self, m):
        return self.X - m

    def K(self, X1, X2=None):
        return tf.stack([model.kern.K(X1, X2) for model in self.models])

    @property
    def Y(self):
        return tf.concat([model.Y.parameter_tensor for model in self.models],
                         axis=1)

    @property
    def X(self):
        return self.models[0].X.parameter_tensor

    @property
    def lengthscales(self):
        return tf.stack([
            model.kern.lengthscales.constrained_tensor for model in self.models
        ])

    @property
    def variance(self):
        return tf.stack(
            [model.kern.variance.constrained_tensor for model in self.models])

    @property
    def noise(self):
        return tf.stack([
            model.likelihood.variance.constrained_tensor
            for model in self.models
        ])
