import environment as env
import trajectory as tj
import functions as fn
import numpy as np
from matplotlib import pyplot as plt


def get_max_action(Q, state):
    """
    give the action that provides the maximum of Q for a state along all the actions
    """
    score = []
    actions = []

    # look over all actions to determine the most profitable
    for u in env.action_space:
        score.append(Q[(state, u)])
        actions.append(u)

    # look which action gives best reward
    index = score.index(max(score))

    return actions[index]


def get_max_value(Q, state):
    """
    give the maximum of Q for a state along all the actions
    """
    values = []
    for action in env.action_space:
        values.append(Q[(state, action)])

    return max(values)


def display(Q_learning, qN=3):
    """
    display the maximum difference between Q_learning and Q
    """

    # determine Q using the optimal policy
    Q = fn.optimal_policy(qN)

    diff = []

    # we compute the infinite norm, i.e. the maximum difference between Q_learning and Q of all the states
    for state in env.state_space:
        value = []

        for action in env.action_space:
            ql = Q_learning[(state, action)]
            q = Q[(state, action)]
            value.append(round(abs(ql - q), 2))

        diff.append(max(value))

    error = max(diff)
    print("     ||Q_learning - Q||inf = " + str(error))

    return error



def protocol_1(discount_factor=0.99, alpha=0.05, epsilon=0.25):
    """
    first experimental protocol
    """
    error = []
    Q = {}

    # initialization
    for x in env.state_space:
        for u in env.action_space:
             # initialize Q to 0 everywhere
            Q[(x, u)] = 0

    for episode in range(100):
        # initial state
        state = (3, 0)

        for transition in range(1000):
            action = None
            p = np.random.default_rng().random()

            # epsilon-greedy policy
            if p < 1 - epsilon:  # exploitation
                action = get_max_action(Q, state)
            else:  # exploration
                action = tj.policy()

            next_state = env.f(state, action)

            # extract reward associates with state x and action u
            reward = env.rewards[next_state[0]][next_state[1]]

            # compute the max value of Q for the state x'
            maxQ = get_max_value(Q, next_state)

            # update Q
            Q[(state, action)] = (1-alpha)*Q[(state, action)] + alpha*(reward + discount_factor*maxQ)

            state = next_state

        print("episode : " + str(episode+1))
        error.append(display(Q))

    return error


def protocol_2(discount_factor=0.99, epsilon=0.25):
    """
    second experimental protocol
    """
    error = []
    Q = {}

    for x in env.state_space:
        for u in env.action_space:
            Q[(x, u)] = 0

    for episode in range(100):
        alpha = 0.05
        state = (3, 0)
        for transition in range(1000):
            action = None
            p = np.random.default_rng().random()

            if p < 1 - epsilon:
                action = get_max_action(Q, state)
            else:
                action = tj.policy()

            next_state = env.f(state, action)
            reward = env.rewards[next_state[0]][next_state[1]]
            maxQ = get_max_value(Q, next_state)

            Q[(state, action)] = (1-alpha)*Q[(state, action)] + alpha*(reward + discount_factor*maxQ)  # update Q

            state = next_state
            alpha *= 0.8

        print("episode : " + str(episode+1))
        error.append(display(Q))

    return error


def protocol_3(discount_factor=0.99, alpha=0.05, epsilon=0.25):
    """
    third experimental protocol
    """
    error = []
    Q = {}

    for x in env.state_space:
        for u in env.action_space:
            Q[(x, u)] = 0

    for episode in range(100):
        state = (3, 0)
        buffer = []
        for transition in range(1000):
            action = None
            p = np.random.default_rng().random()

            if p < 1 - epsilon:
                action = get_max_action(Q, state)
            else:
                action = tj.policy()

            next_state = env.f(state, action)
            reward = env.rewards[next_state[0]][next_state[1]]

            # add the transition to the buffer
            buffer.append((state, action, reward, next_state))

            # update Q ten times using the buffer
            for count in range(10):
                index = np.random.randint(0, len(buffer))
                x, u, r, next_x = buffer[index]
                maxQ = get_max_value(Q, next_x)

                Q[(state, action)] = (1-alpha)*Q[(state, action)] + alpha*(reward + discount_factor*maxQ)

            state = next_state

        print("episode : " + str(episode+1))
        error.append(display(Q))

    return error


if __name__ == '__main__':
    episode = range(100)
    error_1 = protocol_1()
    error_2 = protocol_2()
    error_3 = protocol_3()

    fig, axs = plt.subplots(1, 1)
    axs.plot(episode, error_1, label='protocol 1')
    axs.plot(episode, error_2, label='protocol 2')
    axs.plot(episode, error_3, label='protocol 3')
    axs.set_ylabel('|| $\hat{Q}$ - $Q$ ||$_\infty$')
    axs.set_xlabel('episode')
    axs.set_title('Convergence of $\hat{Q}$ to $Q$')
    axs.legend()
    plt.show()
