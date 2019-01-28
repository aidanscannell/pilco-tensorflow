# import math
# import numpy as np
#
# odei = [1, 2, 3, 4]  #% varibles for the ode solver
# augi = []  #% variables to be augmented
# dyno = [1, 2, 3, 4]  #% variables to be predicted (and known to loss)
# angi = [4]
# #% angle variables
# dyni = [1, 2, 3, 5, 6]  #% variables that serve as inputs to the dynamics GP
# poli = [1, 2, 3, 5, 6]  #% variables that serve as inputs to the policy
# difi = [1, 2, 3, 4]  #% variables that are learned via differences
#
# # 2. Set up the scenario
# dt = 0.10  # [s] sampling time
# T = 4.0  # [s] initial prediction horizon time
# H = math.ceil(T / dt)  # prediction steps (optimization horizon)
# mu0 = np.zeros([4, 1])  # initial state mean
# S0 = np.diag(np.ones(4) * 0.1**2)  # initial state covariance
# N = 15  # number controller optimizations
# J = 1  # initial J trajectories of length H
# K = 1  # no. of initial states for which we optimize
# nc = 10  # number of controller basis functions
#
# # 3. Plant structure
# # plant_dynamics = @dynamics_cp;                    # dynamics ode function
# plant_noise = np.diag(np.ones(4) * 0.01**2)  # measurement noise
# plant_dt = dt
# # plant_ctrl = @zoh                                # controler is zero order hold
# plant_odei = odei
# plant_augi = augi
# plant_angi = angi
# plant_poli = poli
# plant_dyno = dyno
# plant_dyni = dyni
# plant_difi = difi
# # plant_prop = @propagated;
#
# # 4. Policy structure
# # policy.fcn = @(policy,m,s)conCat(@congp,@gSat,policy,m,s);% controller
# #                                                           % representation
# maxU = [10]  # max. amplitude of control
# # [mm ss cc] = gTrig(mu0, S0, plant.angi);                  % represent angles
# mm = [[mu0], [mm]]
# cc = S0 * cc
# ss = [[S0, cc], [cc.T, ss]]  # in complex plane
# # policy_inputs = gaussian(mm(poli), ss(poli,poli), nc)';  # init. location of basis functions
# policy_inputs = np.random.multivariate_normal(
#     mean, cov, nc)  # init. location of basis functions
# policy_targets = 0.1 * np.random.randn(
#     nc, len(maxU))  # init. policy targets (close to zero) N(0, 0.1^2)
# policy_hyp = np.log([[1], [1], [1], [0.7], [0.7], [1],
#                      [0.01]])  # initialize policy hyper-parameters
#
# # 5. Set up the cost structure
# # cost.fcn = @loss_cp;                       % cost function
# cost_gamma = 1  # discount factor
# cost_p = 0.5  # length of pendulum
# cost_width = 0.25  # cost function width
# cost_expl = 0.0  # exploration parameter (UCB)
# cost_angle = plant_angi  # index of angle (for cost function)
# cost_target = np.array([[0], [0], [0], [np.pi]])  # target state
#
# # 6. Dynamics model structure
# # dynmodel_fcn = @gp1d;                # function for GP predictions
# # dynmodel_train = @train;             # function to train dynamics model
# dynmodel_induce = np.zeros([300, 0])  # shared inducing inputs (sparse GP)
# trainOpt = [
#     300, 500
# ]  # defines the max. number of line searches when training the GP dynamics models
# # trainOpt(1): full GP,
# # trainOpt(2): sparse GP (FITC)
#
# # 7. Parameters for policy optimization
# opt_length = 150  # max. number of line searches
# opt_MFEPLS = 30  # max. number of function evaluations
# # per line search
# opt_verbosity = 1  # verbosity: specifies how much information is displayed during policy learning. Options: 0-3
