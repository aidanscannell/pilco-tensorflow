import numpy as np
import tensorflow as tf
import time
import gpflow
from IPython.display import display

from pilco.models.mgpr import MGPR
from pilco.models.sparse_mgpr import SMGPR

float_type = gpflow.settings.dtypes.float_type


class PILCO(gpflow.models.Model):
    def __init__(self,
                 x_train,
                 y_train,
                 policy,
                 cost,
                 horizon=30,
                 m_init=None,
                 s_init=None,
                 name="PILCO"):
        """
        
        Args:
            x_train: Training input data (N x state_dim)
            y_train: Training target data (N x control_dim) - difference in state
            policy: 
            cost: 
            horizon: Number of timesteps to simulate a policy on the learned dynamics model (before improving policy)
            m_init: 
            s_init: 
            name: 
            
        """
        super(PILCO, self).__init__(name)
        self.state_dim = y_train.shape[1]

        self.control_dim = x_train.shape[1] - y_train.shape[1]
        self.horizon = horizon

        # If no initial state distribution for the rollouts is provided then use the first state in the data set
        if m_init is None or s_init is None:
            self.m_init = x_train[0:1, 0:self.state_dim]
            self.s_init = np.diag(np.ones(self.state_dim) * 0.1)
        else:
            self.m_init = m_init
            self.s_init = s_init

        self.dynamics_model = MGPR(x_train, y_train)
        # self.dynamics_model = SMGPR(x_train, y_train, 200)
        self.policy = policy

        self.reward = cost

    @gpflow.name_scope('likelihood')
    def _build_likelihood(self):
        reward = self.predict(
            m_x=self.m_init, s_x=self.s_init, horizon=self.horizon)[2]
        return reward

    def optimize(self):
        print("Optimizing")
        start = time.time()
        self.dynamics_model.optimize()
        end = time.time()
        print("Optimized GP's for dynamics model in %.1f s" % (end - start))

        start = time.time()
        optimizer = gpflow.train.ScipyOptimizer(options={'maxfun': 500})
        optimizer.minimize(self, disp=True)
        end = time.time()
        print("Finished with Controller's optimization in5%.1f seconds" %
              (end - start))

        print('\n-----Learned models------')
        for model in self.dynamics_model.models:
            display(model)

    @gpflow.autoflow((float_type, [None, None]))
    def compute_action(self, m_x):
        """Returns the control (action) according to the learnt policy
        
        Args:
            m_x: mean of state (1 x state_dim)

        Returns:
            Action (or control) determined by policy
            
        """

        return self.policy.compute_action(
            m_x, tf.zeros([self.state_dim, self.state_dim], float_type))[0]

    def predict(self, m_x, s_x, horizon):
        """Simulates the policy (for a given horizon) on the learnt dynamics model and calculates the sum of the immediate
        costs (value function)
        
        Args:
            m_x: mean of state (1 x state_dim)
            s_x: variance of state (state_dim x state_dim)
            horizon: number of timesteps to simulate

        Returns: 
            End state mean (m_x) and variance (s_x) and the value function
            
        """
        loop_vars = [
            tf.constant(0, tf.int32), m_x, s_x,
            tf.constant([[0]], float_type)
        ]  # [j=0, m_x, s_x, reward=0]

        c = lambda j, m_x, s_x, reward: j < horizon
        b = lambda j, m_x, s_x, reward: (
            j + 1,
            *self.propagate(m_x, s_x),
            tf.add(reward, self.reward.compute_cost(m_x, s_x)[0])
        )
        _, m_x, s_x, reward = tf.while_loop(c, b, loop_vars)

        return m_x, s_x, reward

    def propagate(self, m_x, s_x):
        """Computes the successor state distribution by:
            1. Computing the distribution over actions p(u) ~ N(m_u, s_u) - querying policy with p(x) ~ N(m_x, s_x)
            2. Computing the joint (state/control) distribution p(x, u)
            3. Predicting the change of state distribution p(∆x_{t-1}) using the learnt GP dynamics model
            3. Calculating the successor state distribution p(x_{t}) = p(x_{t-1}) + p(∆x_{t-1})
        
        Args:
            m_x: mean of state (1 x state_dim)
            s_x: variance of state (state_dim x state_dim) 

        Returns: 
            Mean (M_x) and variance (S_x) of the successor state
            
        """

        # p(u_{t-1}) = p(π(x_{t-1}))
        m_u, s_u, c_xu = self.policy.compute_action(m_x, s_x)

        # p(x_{t-1}, u_{t-1})
        m = tf.concat([m_x, m_u], axis=1)
        s1 = tf.concat([s_x, s_x @ c_xu], axis=1)
        s2 = tf.concat([tf.transpose(s_x @ c_xu), s_u], axis=1)
        s = tf.concat([s1, s2], axis=0)

        # Predict p(∆x_{t-1}) using the learnt GP dynamics model
        m_dx, s_dx, c_dx = self.dynamics_model.predict_on_noisy_inputs(m, s)

        # p(x_{t})
        M_x = m_dx + m_x
        S_x = s_dx + s_x + s1 @ c_dx + tf.transpose(c_dx) @ tf.transpose(s1)

        M_x.set_shape([1, self.state_dim])
        S_x.set_shape([self.state_dim, self.state_dim])
        return M_x, S_x
