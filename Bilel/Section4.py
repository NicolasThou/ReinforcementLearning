import numpy as np
from numpy.random import Generator
import environment as env
import functions as fn


def policy(x):
    """
    define a uniform policy
    """
    i = np.random.randint(low=0, high=4)
    actions = list(env.action_space.keys())
    return actions[i]


def create_trajectory(initial_x, T):
    """
    creates ht (with ressources limitation algorithm)
    """
    trajectory = []
    N = {}
    R = {}
    Nx = {}
    p = {}
    r = {}

    for x in env.state_space:  # initializations for r and p computing
        for u in env.action_space:
            N[(x, u)] = 0
            R[(x, u)] = 0
            r[(x, u)] = -1000
            for x0 in env.state_space:
                Nx[(x, u, x0)] = 0
                p[(x, u, x0)] = 0

    x = initial_x
    for i in range(T):  # random trajectory computing
        u = policy(x)
        new_x = env.f(x, u)
        rew = env.rewards[new_x[0]][new_x[1]]
        trajectory.append([x, u, rew, new_x])  # add current information to trajectory history

        N[(x, u)] += 1
        Nx[(x, u, new_x)] += 1
        R[(x, u)] += rew
        r[(x, u)] = R[(x, u)] / N[(x, u)]  # mean of all the rewards
        p[(x, u, new_x)] = Nx[(x, u, new_x)] / N[(x, u)]  # probability reaching state new_x with (x,u)

        x = new_x

    return trajectory, p, r


def J_N(x, mu, r, N, discount_factor=0.99):
    """
    computes J state-value recurrence function
    """
    if N < 0:
        print("N cannot be negative !")
        exit()
    elif N == 0:
        return 0
    else:
        return r[(x, mu[x])] + discount_factor*J_N(env.f(x, mu[x]), mu, r, N-1)


def computes_Q(p, r, N, discount_factor=0.99):
    """
    computes Q state-action value recurrence function
    """
    if N < 0:
        print("N can't be negative")
        return None
    else:
        Q = {}
        for x in env.state_space:
            for u in env.action_space:
                Q[(x, u)] = fn.Q_N(p, r, x, u, N)
        return Q


def compute_difference_along_T(qN):
    # extract a history
    history, p, r = create_trajectory((3, 0), 9000)

    # derive mu* from this history
    Q = computes_Q(p, r, qN)
    old_u = fn.determine_optimal_policy_from_Q(Q)

    diff = []
    for t in range(9200, 10000, 200):
        # extract a new history, longer than previous one
        history, p, r = create_trajectory((3, 0), t)

        # determine mu* according to this history
        Q = computes_Q(p, r, qN)
        u_star = fn.determine_optimal_policy_from_Q(Q)

        count = 0

        # count the number of actions different in those policies
        for old, new in zip(old_u.values(), u_star.values()):
            if old != new:
                count += 1
        count /= 25  # normalize the difference (gives a pourcentage of difference)
        diff.append(count)
        old_u = u_star
    print(diff)


if __name__ == '__main__':
    jN = 299
    qN = 2
    T = 500

    history, p, r = create_trajectory((3, 0), T)
    Q = computes_Q(p, r, qN)
    u_star = fn.determine_optimal_policy_from_Q(Q)

    for x in env.state_space:
        j = round(J_N(x, u_star, r, jN), 2)
        print("x = ", x, " | N = ", jN, " | J(N,x) = ", j)
