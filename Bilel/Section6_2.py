import environment as env
import trajectory as tj
import functions as fn
import numpy as np


def get_max_action(Q, state):
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
    values = []
    for action in env.action_space:
        values.append(Q[(state, action)])

    return max(values)


def display(Q_learning, qN=2):
    optimal_policy = fn.optimal_policy(qN)

    for N in range(50, 500, 50):
        diff = []
        for state in env.state_space:
            value = []
            for action in env.action_space:
                value.append(Q_learning[(state, action)])

            q = max(value)
            j = fn.J_N(state, optimal_policy, N)

            diff.append(round(abs(q - j), 2))

        print("     ||Q - J||inf = " + str(max(diff)))
        print()


def protocol_1(discount_factor=0.99, alpha=0.05, epsilon=0.25):
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
            if p < 1 - epsilon:
                action = get_max_action(Q, state)
            else:  # random action
                action = tj.policy()

            next_state = env.f(state, action)
            reward = env.rewards[next_state[0]][next_state[1]]
            maxQ = get_max_value(Q, next_state)

            Q[(state, action)] = (1-alpha)*Q[(state, action)] + alpha*(reward + discount_factor*maxQ)  # update Q

            state = next_state

        print("episode : " + str(episode+1))
        display(Q)


def protocol_2(discount_factor=0.99, epsilon=0.25):
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
        display(Q)


def protocol_3(discount_factor=0.99, alpha=0.05, epsilon=0.25):
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
        display(Q)


if __name__ == '__main__':
    protocol_1()
