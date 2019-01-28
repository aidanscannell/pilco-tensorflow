import gym
import numpy as np
from pilco.models.pilco import PILCO
from pilco.policies import RBFNPolicy
from pilco.cost import SaturatingCost

# from pilco.cost import ExponentialReward

from env.cart_pole_env import CartPole
env = CartPole()

# env = gym.make('InvertedPendulum-v2')
env.reset()


def rollout(policy, timesteps):
    """ Performs a rollout of a given policy on the environment for a given number of timesteps
    Args:
        policy: function representing policy to rollout
        timesteps: the number of timesteps to perform

    Returns:
        X: list of training inputs - (x, u) where x=state and u=control
        Y: list of training targets - differences y = x(t) - x(t-1)

    """
    x_set = []
    y_set = []
    env.reset()
    x, _, _, _ = env.step(0)
    x = np.append(x, np.sin(x[1]))
    x = np.append(x, np.cos(x[1]))
    x = np.delete(x, 1)
    print(x)
    for t in range(timesteps):
        env.render()
        # print(x)
        u = policy(x)
        # print("u_%i: %.3f" % (t, u))
        x_new, reward, done, _ = env.step(u)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
        x_new = np.append(x_new, np.sin(x_new[1]))
        x_new = np.append(x_new, np.cos(x_new[1]))
        x_new = np.delete(x_new, 1)
        x_set.append(np.hstack((x, u)))
        y_set.append(x_new - x)
        x = x_new
    return np.stack(x_set), np.stack(y_set)


def random_policy(x):
    # a = env.action_space.sample() * 4
    a = env.action_space.sample()
    # if a >= env.action_space.high:
    #     a = env.action_space.high
    # if a <= env.action_space.low:
    #     a = env.action_space.low
    print(a)
    return a
    # return env.action_space.sample()


def pilco_policy(x):
    return pilco.compute_action(x[None, :])[0, :]


x, y = rollout(policy=random_policy, timesteps=40)
for i in range(1, 3):
    x_, y_ = rollout(policy=random_policy, timesteps=40)
    x = np.vstack((x, x_))
    y = np.vstack((y, y_))

state_dim = y.shape[1]
control_dim = x.shape[1] - state_dim

RBFNPolicy = RBFNPolicy(
    state_dim,
    control_dim=control_dim,
    num_basis_fun=10,
    max_action=env.action_space.high[0])

a = 0.25
l = 0.6
C = np.array([[1., l, 0.0], [0., 0., l]])
iT = a**(-2) * np.dot(C.T, C)

cost = SaturatingCost(state_dim=3, state_idxs=[0, 3, 4], W=iT, t=[0., 0., -1.])
# print(state_dim)
# cost = SaturatingCost(state_dim=state_dim)
# cost = SaturatingCost(
#     x_target=[0., 0., -1.], iT=iT, state_dim=3, state_idxs=[0, 3, 4])
# cost = ExponentialReward(state_dim)

pilco = PILCO(x, y, policy=RBFNPolicy, cost=cost, horizon=40)

for rollouts in range(15):
    # Learn dynamics model & use it to simulate/optimise policy
    pilco.optimize()
    import pdb

    pdb.set_trace()

    # Execute policy on environment
    x_new, y_new = rollout(policy=pilco_policy, timesteps=100)

    # Update dataset
    x = np.vstack((x, x_new))
    y = np.vstack((y, y_new))
    pilco.dynamics_model.set_XY(x, y)
