import numpy as np
from matplotlib import pyplot as plt
import random
from section3 import *
from section2 import *

"""

1) Implement a routine generated from a random uniform policy which computes p(x0|x, u) and r(x, u) that define 
the equivalent MDP of the domain, together with γ and a trajectory of size T. 
done

2) Afterwards, implement a routine which computes Q(x, u) from these components using the dynamic programming principle. 
done

3) Determine the smallest T such that a greater value does not change the policy inferred from Q. 
not done

4) Derive directly µ∗ from Q.
done 
 
5) Display J_N_µ∗ for each state x.


"""


experience = np.load('./npy/historic.npy', allow_pickle=True)
experience2 = np.load('./npy/historic2.npy', allow_pickle=True)


def estimation_r(x, u):
    """
    This function compute from a random uniform policy an estimation of r(x, u)

    Argument:
    ========
    x : is the state
    u : is the action

    Return:
    ======
    Return an integer which represent the estimation of the value of r(x, u)
    """
    new_state = next_state(domain, x, u)
    return domain[new_state[0], new_state[1]]


def estimation_p(x1, u, x2):
    """
    This function compute from a random uniform policy an estimation of p(x2|x1, u)

    Argument:
    ========
    x1 : is the state before taking the action u
    u : is the action
    x2 : is the state after taking the action u

    Return:
    ======
    Return an integer which represent the estimation of the value of p(x2|x1, u) i-e
    return 0 if it is impossible or 0.5

    """
    size = np.shape(domain)
    n, m = size[0], size[1]
    x = min(max(x1[0] + u[0], 0), n - 1)
    y = min(max(x1[1] + u[1], 0), m - 1)
    new_state = [x, y]
    if x2 == new_state or x2 == [0, 0]:
        return 0.5
    else:
        return 0


def state_space(x):
    """
    Initialize the state space of every state that are possible to reach from x according to the action space

    Return:
    ======
    return a list of list. So a list of state.
    """
    X = []
    for action in a:
        X.append(next_state(domain, x, action))
    return X


def state_space2(h):
    """
    Initialize the state space of every state that our agent has met

    Argument:
    ========
    h is the experience

    Return:
    ======
    return a list of list. So a list of state.
    """
    space = []
    for i in range(0, len(h), 3):
        if (h[i] in space) is False:
            space.append(h[i])
    return space


def q_function(x, u, n):
    """
    Computes Q(x, u) from the estimation of p(x2|x1, u) and the estimation of r(x, u) using the dynamic programming
    principle. We remind that γ is equal to 0.99 in this assignment. We remind that 'a' is the action space

    Argument:
    ========
    x : is the state
    u : is the action
    n : is the number of the iteration of the Q_function

    Return:
    =======
    return an integer which represent the result of the Q function i-e the state-action value function.
    """

    if n == 0:
        return 0
    else:
        r = estimation_r(x, u)  # We estimate the reward according the x and u we've ever met
        sum = 0
        X = state_space(x)  # We update state space that it's possible to reach from x
        for i in X:
            list_q = []  # we try to find the max in this list
            p = estimation_p(x, u, i)
            for action in a:
                list_q.append(q_function(i, action, n-1))
            value_max = max(list_q)
            sum += p * value_max
        return r + (0.99 * sum)


def compute_policy(n):
    """
    This function estimate for each state, the best action to do.

    Argument:
    ========
    n : is the number of the iteration of the Q_function

    Return:
    =======
    the space action which is the corresponding best action for each state (output corresponding to the input of the policy)

    """
    space = []
    for i in range(5):
        state = []
        state.append(i)
        for j in range(5):
            state.append(j)
            best_q_value = []
            for action in a:
                best_q_value.append(q_function(state, action, n))
            v_max = max(best_q_value)
            indice = best_q_value.index(v_max)
            space.append(a[indice])

    return space


optimal_policy = compute_policy(4)


def final_policy_section4(x):
    """
    return the action to do according to a state.

    Argument:
    ========
    x is a state

    Return:
    =======
    return an action
    """
    return optimal_policy[x[0] * 5 + x[1]]


def value_function_section4(x, n):
    """
    Compute the return of a stationary policy thanks to the Bellman Equation
    We use the policy that we found to compute each J(x)
    The discount factor gamma is equal to 0.99

    Argument:
    ========
    x : is the initial state
    n : number of value function

    Return:
    =======
    return an integer, which is the estimation of J^µ

    """
    if n == 0:
        return 0
    else:
        action = final_policy_section4(x)  # here we use the policy
        new_state = next_state(domain, x, action)  # here there is disturbance !
        reward = estimation_r(x, action)
        return reward + 0.99 * value_function_section4(new_state, n-1)


def display_section4(step):
    """
    Display J_N_µ(x) for each state x
    """
    x = [0, 0]
    n, m = np.shape(domain)
    tableau = []  # table of J_N_µ(x) for each state x
    for i in range(n):
        row = []
        for j in range(m):
            x[0] = i
            x[1] = j
            row.append(value_function_section4(x, step))  # here we add the value_function for the state x
        tableau.append(row)

    tableau = np.array(tableau)
    print(tableau)


if __name__ == '__main__':

    print('---------here the domain/environment---------')
    print(domain)  # domain on what we work
    print('--------- here the action space for the policy ---------')
    print(optimal_policy)
    print('--------- test of the policy and its return ---------')
    print(final_policy_section4([3, 4]))
    print('-------- J_N_µ(x) for each state x ---------')
    display_section4(600)






