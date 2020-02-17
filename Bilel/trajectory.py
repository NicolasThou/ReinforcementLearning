import environment as env
import numpy as np


def policy():  # return a random action
    i = np.random.randint(low=0, high=4)
    actions = list(env.action_space.keys())
    return actions[i]


def create_trajectory(initial_x, T):  # creates ht (with ressources limitation algorithm)
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
        u = policy()
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
