import Bilel.section2 as S2
import Bilel.Section4 as S4
from matplotlib import pyplot as plt
from random import shuffle


def Q_learning(T, alpha=0.05, discount_factor=0.99):
    Q = {}
    for x in S2.state_space:
        for u in S2.action_space:
            # initialize Q to 0 everywhere
            Q[(x, u)] = 0

    trajectory = S4.create_trajectory((0, 3), T)
    shuffle(trajectory)

    for sample in trajectory:
        state, action, reward, new_state = sample

        # determine max of Q(x(t+1))
        maxQ = int('-inf')
        for u in S2.action_space:
            maxQ = Q[(new_state, u)] if Q[(new_state, u)] > maxQ else maxQ

        Q[(state, action)] = (1 - alpha)*Q[(state, action)] + alpha*(reward + discount_factor*maxQ)  # update

    return Q


def determine_optimal_policy_from_Q_leanring(Q):  # determine the optimal policy
    opt_policy = {}

    # determine best action for each state according to Qn
    for x in S2.state_space:
        #print(x)  # used to know where the program is (this function is actually long depending on N)
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


def J_N(x, mu, N, discount_factor=0.99):  # computes J state-value recurrence function with policy µ
    if N < 0:
        print("N cannot be negative !")
        exit()
    elif N == 0:
        return 0
    else:
        new_state = S2.f(x, mu[x])
        return S2.rewards[new_state[0]][new_state[1]] + discount_factor*J_N(new_state, mu, N-1)


if __name__ == '__main__':
    Q = Q_learning(300)
    mu_learning = determine_optimal_policy_from_Q_leanring(Q)
