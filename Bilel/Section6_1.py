from random import shuffle
import environment as env
import trajectory as tj
import functions as fn


def Q_learning(T, alpha=0.05, discount_factor=0.99):
    """
    Q-learning algorithm
    """
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
            # store the value of all the actions for the next state to extract the max after
            next_state_Q.append(Q[(next_state, u)])

        # update
        Q[(state, action)] = (1 - alpha)*Q[(state, action)] + alpha*(reward + discount_factor*max(next_state_Q))

    return Q


if __name__ == '__main__':
    jN = 100

    # compute the Q-learning algorithm
    Q = Q_learning(1000)

    # infer a policy from Q
    policy_learning = fn.determine_optimal_policy_from_Q(Q)

    # extract the optimal policy
    optimal_policy = fn.optimal_policy(4)

    # computation of J_optimal and J_learning
    for x, i in zip(env.state_space, range(len(env.state_space))):
        # J_learning is computed using the policy inferred from the Q-learning
        j_learning = round(fn.J_N(x, policy_learning, jN), 2)
        j_optimal = round(fn.J_N(x, optimal_policy, jN), 2)

        diff = round(abs(j_learning - j_optimal), 2)
        str_x = "x = " + str(x) + " | J_optimal = " + str(j_learning) + " | J_learning = " + str(j_optimal) + " | diff = " + str(diff)
        print(str_x)
