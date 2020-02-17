import Section2 as S2
import Section4 as S4
from matplotlib import pyplot as plt


def convergence_speed():
    p_error = []
    r_error = []
    T_values = [i for i in range(100, 1200, 50)]

    for T in T_values:
        history, p, r = S4.create_trajectory((3, 0), T)
        p_sum = 0
        r_sum = 0
        for x in S2.state_space:
            for u in S2.action_space:
                new_state = S2.f(x, u)
                p_sum += 1 - p[(x, u, new_state)]
                r_sum += abs(r[(x, u)] - S2.rewards[new_state[0]][new_state[1]])

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


# compare the state-action value function for an optimal policy inferred from approximations of p and r and
# an optimal policy inferred from the exact one
def compare_optimal_policies(T, qN, jN):
    # compute exact optimal policy
    p = {}  # exact probability
    r = {}  # exact reward
    for x in S2.state_space:
        for u in S2.action_space:
            for next_state in S2.state_space:
                p[(x, u, next_state)] = 0

            new_state = S2.f(x, u)
            p[(x, u, new_state)] = 1
            r[(x, u)] = S2.rewards[new_state[0]][new_state[1]]

    # compute the exact optimal policy
    Q = S4.computes_Q(p, r, qN)
    u_star = S4.determine_optimal_policy(Q)

    # compute approximation of optimal policy
    history, p_appr, r_appr = S4.create_trajectory((3, 0), T)
    Q_appr = S4.computes_Q(p_appr, r_appr, qN)
    u_star_appr = S4.determine_optimal_policy(Q_appr)

    # computation of J_optimal and J_approximate
    J = []
    for x in S2.state_space:
        j_optimal_policy = round(S4.J_N(x, u_star, r, jN), 2)
        j_appr_optimal_policy = round(S4.J_N(x, u_star_appr, r, jN), 2)
        J.append([j_optimal_policy, j_appr_optimal_policy])

    for x, i in zip(S2.state_space, range(len(S2.state_space))):
        diff = abs(J[i][0] - J[i][1])
        str_x = "x = " + str(x) + " | J_optimal = " + str(J[i][0]) + " | J_approximate = " + str(J[i][1]) + " | diff = " + str(diff)
        print(str_x)

    return J


# display the difference between the Q calculated with the MDP structure
# and the Q calculated with the exact p and r
def influence_of_T_on_Q(T, N):
    p = {}  # exact probability
    r = {}  # exact reward
    for x in S2.state_space:
        for u in S2.action_space:
            new_state = S2.f(x, u)
            p[(x, u, new_state)] = 1
            r[(x, u)] = S2.rewards[new_state[0]][new_state[1]]

    Q = {}
    for x in S2.state_space:
        for u in S2.action_space:
            history, p_appr, r_appr = S4.create_trajectory((3, 0), T)
            Q_exact = round(S4.Q_N(p, r, x, u, N), 2)
            Q_appr = round(S4.Q_N(p_appr, r_appr, x, u, N), 2)
            Q[(x, u)] = (Q_exact, Q_appr)

    for key in Q:
        diff = abs(Q[key][0] - Q[key][1])
        str_x = "(x,u) = " + str(key) + " | Q_exact = " + str(Q[key][0]) + " | Q_appr = " + str(Q[key][1]) + " | diff = " + str(diff)
        #print(str_x)

    return Q


if __name__ == '__main__':
    # T, p_error, r_error = convergence_speed()
    # J = compare_optimal_policies(400, 3, 100)
    T = [t for t in range(100, 1100, 100)]
    error = []
    for t in T:
        print(t)
        Q = influence_of_T_on_Q(t, 3)
        sum = 0
        for key in Q:
            sum += abs(Q[key][0] - Q[key][1])

        error.append(sum)

    plt.plot(T, error)
    plt.show()
