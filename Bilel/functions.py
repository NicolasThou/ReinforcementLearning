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
            next_state = env.f(state, u)

            Qs = []

            # store the reward recording for each action
            for u1 in env.action_space.keys():
                Qs.append(Q_N(p, r, next_state, u1, N-1))

            # look for which action gives best reward
            max_Q = max(Qs)

            # actualize the sum term in Qn recurrence formula
            sum_Q += p[(state, action, next_state)]*max_Q
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
