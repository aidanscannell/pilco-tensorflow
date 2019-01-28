import numpy as np

# import matlab.engine
#
# eng = matlab.engine.start_matlab()


def loss_cp(m, s):
    cw = 0.25  # cost function width
    b = 0.0  # exploration parameter

    state_dim = m.shape[0]
    D0 = state_dim  # state dimension
    D1 = state_dim + 2  # state dimension with cos/sin

    M = np.zeros([D1, 1])
    M[0:D0, 0:1] = m
    S = np.zeros([D1, D1])
    S[0:D0, 0:D0] = s

    Mdm = np.concatenate([np.eye(D0), np.zeros([D1 - D0, D0])])
    Sdm = np.zeros([D1 * D1, D0])
    Mds = np.zeros([D1, D0 * D0])
    Sds = np.kron(Mdm, Mdm)

    # 2.Define static penalty as distance from target setpoint
    ell = 0.6  # pendulum length
    Q = np.zeros([D1, D1])
    Q[[[0, 0], [D0, D0]], [[0, D0], [0, D0]]] = np.array(
        [[1], [ell]]) @ np.array([[1, ell]])
    Q[D0 + 1, D0 + 1] = ell**2

    # 3. Augment angles????

    # # 4. Calculate loss!
    # L = 0
    # dLdm = np.zeros([1, D0])
    # dLds = np.zeros([1, D0 * D0])
    # S2 = 0
    #
    # for i in range(1, len(cw)): # scale mixture of immediate costs
    #     cost.z = target;
    #     cost.W = Q / cw(i) ^ 2;
    #
    #
    #     [r rdM rdS s2 s2dM s2dS] = lossSat(cost, M, S);
    #
    #     L = L + r;
    #     S2 = S2 + s2;
    #     dLdm = dLdm + rdM(:)'*Mdm + rdS(:)' * Sdm;
    #     dLds = dLds + rdM(:)'*Mds + rdS(:)' * Sds;
    #
    #     if (b~=0 | | ~isempty(b)) & & abs(s2) > 1e-12
    #     L = L + b * sqrt(s2);
    #     dLdm = dLdm + b / sqrt(s2) * (s2dM(:)
    #     '*Mdm + s2dS(:)' * Sdm ) / 2;
    #     dLds = dLds + b / sqrt(s2) * (s2dM(:)
    #     '*Mds + s2dS(:)' * Sds ) / 2
    return


m = np.array([[1, 2, 3, 4]]).T

s = 0.1 * np.ones([4, 4])

loss_cp(m, s)
