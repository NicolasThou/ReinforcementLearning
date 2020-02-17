import environment as env


def Q_N(p, r, state, action, N, discount_factor=0.99):  # computes Q state-action value recurrence function
    if N < 0:
        print("N can't be negative")
        return None
    elif N == 0:
        return 0
    else:
        sum_Q = 0

        for u in env.action_space:
            # we are only looking for the state which are 'reachable' from x since others one will have p(x'|x,u)=0
            x = env.f(state, u)
            Qs = []

            # store the reward recording for each action
            for u1 in env.action_space:
                Qs.append(Q_N(p, r, x, u1, N-1))

            # look for which action gives best reward
            max_Q = max(Qs)

            # actualize the sum term in Qn recurrence formula
            sum_Q += p[(state, action, x)]*max_Q
        return r[(state, action)] + discount_factor*sum_Q


def J_N(x, mu, N, discount_factor=0.99):  # computes J state-value recurrence function with policy µ
    if N < 0:
        print("N cannot be negative !")
        return None
    elif N == 0:
        return 0
    else:
        new_state = env.f(x, mu[x])
        return env.rewards[new_state[0]][new_state[1]] + discount_factor*J_N(new_state, mu, N-1)


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


def optimal_policy(N):
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
            Q[(x, u)] = Q_N(p, r, x, u, N)

    return determine_optimal_policy_from_Q(Q)
