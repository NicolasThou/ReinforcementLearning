from matplotlib import pyplot as plt
import environment as env
import trajectory as tj
import functions as fn


def convergence_speed():
    p_error = []
    r_error = []
    T_values = [i for i in range(100, 1200, 50)]

    for T in T_values:
        history, p, r = tj.create_trajectory((3, 0), T)
        p_sum = 0
        r_sum = 0
        for x in env.state_space:
            for u in env.action_space:
                new_state = env.f(x, u)
                p_sum += 1 - p[(x, u, new_state)]
                r_sum += abs(r[(x, u)] - env.rewards[new_state[0]][new_state[1]])

        p_error.append(p_sum)
        r_error.append(r_sum)

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)

    axs[0].plot(T_values, p_error)
    axs[0].set_ylabel('diff (p,$\^p$)')
    axs[0].set_xlabel('T')
    axs[0].set_title('Convergence speed of $\^p$')

    axs[1].plot(T_values, r_error)
    axs[1].set_ylabel('diff (r,$\^r$)')
    axs[1].set_xlabel('T')
    axs[1].set_title('Convergence speed of $\^r$')

    plt.show()
    return T_values, p_error, r_error


def computes_Q(p, r, N, discount_factor=0.99):
    """
    computes Q state-action value recurrence function
    """
    if N < 0:
        print("N can't be negative")
        return None
    else:
        Q = {}
        for x in env.state_space:
            for u in env.action_space:
                Q[(x, u)] = fn.Q_N(p, r, x, u, N)
        return Q


def compare_policies(T, qN, jN):
    """
    compare the state-action value function for an optimal policy inferred from approximations of p and r and
    an optimal policy inferred from the exact one
    """

    # compute optimal exact
    optimal_policy = fn.optimal_policy(qN)

    # compute approximation of optimal policy
    history, p, r = tj.create_trajectory((3, 0), T)
    Q_learned = computes_Q(p, r, qN)
    learned_policy = fn.determine_optimal_policy_from_Q(Q_learned)

    # computation of J_optimal and J_approximate
    J = []
    for x in env.state_space:
        j_optimal = round(fn.J_N(x, optimal_policy, jN), 2)
        j_learned = round(fn.J_N(x, learned_policy, jN), 2)
        J.append([j_optimal, j_learned])

    for x, i in zip(env.state_space, range(len(env.state_space))):
        diff = round(abs(J[i][0] - J[i][1]), 2)
        str_x = "x = " + str(x) + " | J_optimal = " + str(J[i][0]) + " | J_approximate = " + str(J[i][1]) + " | diff = " + str(diff)
        print(str_x)

    return J


def influence_of_T_on_Q(T, N):
    """
    display the difference between the Q calculated with the MDP structure
    and the Q calculated with the exact p and r
    """

    # exact probability
    p = {}

    # exact reward
    r = {}

    for x in env.state_space:
        for u in env.action_space:
            for next_state in env.state_space:
                p[(x, u, next_state)] = 0

            new_state = env.f(x, u)
            p[(x, u, new_state)] = 1
            r[(x, u)] = env.rewards[new_state[0]][new_state[1]]

    Q = {}
    for x in env.state_space:
        for u in env.action_space:
            history, p_appr, r_appr = tj.create_trajectory((3, 0), T)
            Q_optimal = round(fn.Q_N(p, r, x, u, N), 2)
            Q_learned = round(fn.Q_N(p_appr, r_appr, x, u, N), 2)
            Q[(x, u)] = (Q_optimal, Q_learned)

    for key in list(Q.keys()):
        diff = abs(Q[key][0] - Q[key][1])
        str_x = "(x,u) = " + str(key) + " | Q_exact = " + str(Q[key][0]) + " | Q_appr = " + str(Q[key][1]) + " | diff = " + str(diff)

    return Q


if __name__ == '__main__':
    # compute the convergence of p and r
    # T, p_error, r_error = convergence_speed()

    # compare the policy inferred from Q computed using a trajectory to a real optimal policy
    # J = compare_policies(t, 3, 100)

    # display the convergence of Q along T
    T = [t for t in range(100, 1100, 100)]
    error = []
    for t in T:
        print(t)
        Q = influence_of_T_on_Q(t, 3)
        sum = 0
        for key in Q:
            sum += abs(Q[key][0] - Q[key][1])

        error.append(sum)

    # plt.plot(T, error)
    fig, axs = plt.subplots(1, 1)
    axs.plot(T, error)
    axs.set_ylabel('|| $\^Q$ - Q ||$_\infty$')
    axs.set_xlabel('T')
    axs.set_title('Convergence of $\^Q$ to Q')
    plt.show()
