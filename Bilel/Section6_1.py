from random import shuffle
import math
import environment as env
import trajectory as tj
import functions as fn


def Q_learning(T, alpha=0.05, discount_factor=0.99):
    Q = {}

    # initialization
    for x in env.state_space:
        for u in env.action_space:
            # initialize Q to 0 everywhere
            Q[(x, u)] = 0

    # creates and shuffle a new trajectory
    trajectory, ignore, ignore2 = tj.create_trajectory((0, 3), T)
    shuffle(trajectory)

    # update Q using the trajectory
    for sample in trajectory:
        state, action, reward, next_state = sample

        maxQ = -math.inf
        # determine max of Q(x(k+1))
        for u in env.action_space:
            maxQ = Q[(next_state, u)] if Q[(next_state, u)] > maxQ else maxQ

        Q[(state, action)] = (1 - alpha)*Q[(state, action)] + alpha*(reward + discount_factor*maxQ)  # update

    return Q


def determine_optimal_policy_from_Q(Q):  # determine the optimal policy
    policy = {}

    # determine best action for each state according to Q
    for x in env.state_space:
        score = []
        actions = []

        # look over all actions to determine the most profitable
        for u in env.action_space:
            score.append(Q[(x, u)])
            actions.append(u)

        # look which action gives best reward
        index = score.index(max(score))
        best_action = actions[index]

        # save action for this state
        policy[x] = best_action

    return policy


def optimal_policy(qN):
    p = {}  # exact probability
    r = {}  # exact reward

    for x in env.state_space:
        for u in env.action_space:
            for next_state in env.state_space:
                p[(x, u, next_state)] = 0

            new_state = env.f(x, u)
            p[(x, u, new_state)] = 1
            r[(x, u)] = env.rewards[new_state[0]][new_state[1]]

    # compute the exact optimal policy
    Q = {}
    for x in env.state_space:
        for u in env.action_space:
            Q[(x, u)] = fn.Q_N(p, r, x, u, qN)

    u_star = determine_optimal_policy_from_Q(Q)

    return u_star


if __name__ == '__main__':
    T = 1000
    qN = 3
    jN = 200

    Q = Q_learning(T)
    policy_learning = determine_optimal_policy_from_Q(Q)

    optimal_policy = optimal_policy(qN)

    # computation of J_optimal and J_approximate
    J = []
    for x in env.state_space:
        j_learning = round(fn.J_N(x, policy_learning, jN), 2)
        j_optimal = round(fn.J_N(x, optimal_policy, jN), 2)
        J.append([j_learning, j_optimal])

    for x, i in zip(env.state_space, range(len(env.state_space))):
        diff = round(abs(J[i][0] - J[i][1]), 2)
        str_x = "x = " + str(x) + " | J_optimal = " + str(J[i][0]) + " | J_learning = " + str(J[i][1]) + " | diff = " + str(diff)
        print(str_x)
