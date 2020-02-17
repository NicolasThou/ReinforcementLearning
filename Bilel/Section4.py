import Bilel.Section2 as S2
import numpy as np
from numpy.random import Generator


def policy(x):  # define a uniform policy
    i = np.random.randint(low=0, high=4)
    actions = list(S2.action_space.keys())
    return actions[i]


def create_trajectory(initial_x, T):  # creates ht (with ressources limitation algorithm)
    trajectory = []
    N = {}
    R = {}
    Nx = {}
    p = {}
    r = {}

    for x in S2.state_space:  # initializations for r and p computing
        for u in S2.action_space:
            N[(x, u)] = 0
            R[(x, u)] = 0
            r[(x, u)] = -1000
            for x0 in S2.state_space:
                Nx[(x, u, x0)] = 0
                p[(x, u, x0)] = 0

    x = initial_x
    for i in range(T):  # random trajectory computing
        u = policy(x)
        new_x = S2.f(x, u)
        rew = S2.rewards[new_x[0]][new_x[1]]
        trajectory.append([x, u, rew, new_x])  # add current information to trajectory history

        N[(x, u)] += 1
        Nx[(x, u, new_x)] += 1
        R[(x, u)] += rew
        r[(x, u)] = R[(x, u)] / N[(x, u)]  # mean of all the rewards
        p[(x, u, new_x)] = Nx[(x, u, new_x)] / N[(x, u)]  # probability reaching state new_x with (x,u)

        x = new_x

    return trajectory, p, r


def Q_N(p, r, x, u, N, discount_factor=0.99):  # computes Q state-action value recurrence function
    if N < 0:
        print("N can't be negative")
        exit()
    elif N == 0:
        return 0
    else:
        sum_Q = 0

        for u0 in S2.action_space.keys():
            # we are only looking for the state which are 'reachable' from x since others one will have p(x'|x,u)=0
            next_state = S2.f(x, u)

            Qs = []

            # store the reward recording for each action
            for u1 in S2.action_space.keys():
                Qs.append(Q_N(p, r, next_state, u1, N-1))

            # look for which action gives best reward
            max_Q = max(Qs)

            # actualize the sum term in Qn recurrence formula
            sum_Q += p[(x, u, next_state)]*max_Q
        return r[(x, u)] + discount_factor*sum_Q


def determine_optimal_policy(Q):  # determine the optimal policy
    opt_policy = {}

    # determine best action for each state according to Qn
    for x in S2.state_space:
        score = []
        actions = []

        # look over all actions to determine the most profitable
        for u in S2.action_space:
            score.append(Q[(x, u)])
            actions.append(u)

        # look which action gives best reward
        index = score.index(max(score))
        best_action = actions[index]

        # save action for this state
        opt_policy[x] = best_action

    return opt_policy


def J_N(x, mu, r, N, discount_factor=0.99):  # computes J state-value recurrence function
    if N < 0:
        print("N cannot be negative !")
        exit()
    elif N == 0:
        return 0
    else:
        return r[(x, mu[x])] + discount_factor*J_N(S2.f(x, mu[x]), mu, r, N-1)


def computes_Q(p, r, N, discount_factor=0.99):  # computes Q state-action value recurrence function
    if N < 0:
        print("N can't be negative")
        return None
    else:
        Q = {}
        for x in S2.state_space:
            for u in S2.action_space:
                Q[(x, u)] = Q_N(p, r, x, u, N)
        return Q


def compute_difference_along_T(qN):
    # extract a history
    history, p, r = create_trajectory((3, 0), 9000)

    # derive mu* from this history
    Q = computes_Q(p, r, qN)
    old_u = determine_optimal_policy(Q)

    diff = []
    for t in range(9200, 10000, 200):
        # extract a new history, longer than previous one
        history, p, r = create_trajectory((3, 0), t)

        # determine mu* according to this history
        Q = computes_Q(p, r, qN)
        u_star = determine_optimal_policy(Q)

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
    u_star = determine_optimal_policy(Q)

    for x in S2.state_space:
        j = round(J_N(x, u_star, r, jN), 2)
        print("x = ", x, " | N = ", jN, " | J(N,x) = ", j)
