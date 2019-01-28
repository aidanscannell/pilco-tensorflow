from math import pi, ceil
import numpy as np
from examples.g_trig import gTrig


class Structure:
    pass


# 1b. Important indices
# odei  indicies for the ode solver
# augi  indicies for variables augmented to the ode variables
# dyno  indicies for the output from the dynamics model and indicies to loss
# angi  indicies for variables treated as angles (using sin/cos representation)
# dyni  indicies for inputs to the dynamics model
# poli  indicies for the inputs to the policy
# difi  indicies for training targets that are differences (rather than values)

# odei = [1, 2, 3, 4]  # varibles for the ode solver
augi = np.array([])  # variables to be augmented
dyno = np.array([1, 2, 3, 4])  # variables to be predicted (and known to loss)
angi = np.array([1, 4])  # angle variables
dyni = np.array([1, 2, 3, 5,
                 6])  # variables that serve as inputs to the dynamics GP
poli = np.array([1, 2, 3, 5,
                 6])  # variables that serve as inputs to the policy
difi = np.array([1, 2, 3, 4])  # variables that are learned via differences

# 2. Set up the scenario
dt = 0.10  # [s] sampling time
# T = 4.0  # [s] initial prediction horizon time
# H = ceil(T / dt)  # prediction steps (optimization horizon)
H = 40
mu0 = np.zeros([4, 1])  # initial state mean
S0 = np.diag(np.ones(4) * 0.1**2)  # initial state covariance
N = 15  # number controller optimizations
J = 1  # initial J trajectories of length H
K = 1  # no. of initial states for which we optimize
nc = 10  # number of controller basis functions

# 3. Plant structure
plant = Structure()
# plant.dynamics = @dynamics_cp                    # dynamics ode function
plant.noise = np.diag(np.ones(4) * 0.01**2)  # measurement noise
plant.dt = dt
# plant.ctrl = @zoh                                # controler is zero order hold
# plant.odei = odei
plant.augi = augi
plant.angi = angi
plant.poli = poli
plant.dyno = dyno
plant.dyni = dyni
plant.difi = difi
# plant.prop = @propagated;

# 4. Policy structure
policy = Structure()
# policy.fcn = @(policy,m,s)conCat(@congp,@gSat,policy,m,s)# controller representation
policy.maxU = 3  # max. amplitude of control
mm, ss, cc = gTrig(mu0, S0, plant.angi)  # represent angles
mm = np.vstack((mu0, mm))
cc = S0 @ cc
ss = np.vstack((np.hstack((S0, cc)), np.hstack((cc.T,
                                                ss))))  # in complex plane
poli = poli - 1
policy.inputs = np.random.multivariate_normal(
    mm[poli].flatten(), ss[poli[:, np.newaxis], poli],
    nc)  # init. location of basis functions
# policy.p.targets = 0.1*randn(nc, length(policy.maxU))    # init. policy targets (close to zero)
# policy.p.hyp = np.log([[1], [1], [1], [0.7], [0.7], [1], [0.01]])              # initialize policy hyper-parameters

cost = Structure()
# cost.fcn = @loss_cp;                       # cost function
cost.gamma = 1  # discount factor
cost.p = 0.5  # length of pendulum
cost.width = 0.25  # cost function width
cost.expl = 0.0  # exploration parameter (UCB)
cost.angle = plant.angi  # index of angle (for cost function)
cost.target = np.array([[0], [0], [0], [pi]])  # target state
