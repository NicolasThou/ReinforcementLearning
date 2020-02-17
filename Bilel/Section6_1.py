from random import shuffle
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

        next_state_Q = []
        for u in env.action_space:
            next_state_Q.append(Q[(next_state, u)])

        # update
        Q[(state, action)] = (1 - alpha)*Q[(state, action)] + alpha*(reward + discount_factor*max(next_state_Q))

    return Q


if __name__ == '__main__':
    T = 1000
    qN = 3
    jN = 50

    Q = Q_learning(T)
    policy_learning = fn.determine_optimal_policy_from_Q(Q)

    optimal_policy = fn.optimal_policy(qN)

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
