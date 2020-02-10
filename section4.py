import numpy as np
from matplotlib import pyplot as plt
import random
from section2 import *
from section3 import *

"""

1) Implement a routine generated from a random uniform policy which computes p(x0|x, u) and r(x, u) that define 
the equivalent MDP of the domain, together with γ and a trajectory of size T. 

2) Afterwards, implement a routine which computes Q(x, u) from these components using the dynamic programming principle. 

3) Determine the smallest T such that a greater value does not change the policy inferred from Q. 

4) Derive directly µ∗ from Q. Display J_N_µ∗ for each state x.

"""

def estimation_r(trajectory, x, u):
    """
    This function compute from a random uniform policy an estimation of r(x, u) from a trajectory of size T.
    The trajectory is like : (x(0), u(0), r(0), ..., u(t-1), r(t-1), x(t))

    Argument:
    ========
    trajectory : is the historic of each action, move and reward of the agent
    x : is the state
    u : is the action

    Return:
    ======
    Return an integer which represent the estimation of the value of r(x, u)
    """
    estimation = 0
    n = 0
    for i in range(0, len(trajectory)-3, 3):
        if trajectory[i] == x and trajectory[i+1] == u:
            estimation += trajectory[i+2]
            n += 1
    if n == 0:
        return 0
    else:
        return estimation / n


def estimation_p(trajectory, x1, u, x2):
    """
    This function compute from a random uniform policy an estimation of p(x2|x1, u) from a trajectory of size T.
    The trajectory is like : (x(0), u(0), r(0), ..., u(t-1), r(t-1), x(t))

    Argument:
    ========
    trajectory : is the historic of each action, move and reward of the agent
    x1 : is the state before taking the action u
    u : is the action
    x2 : is the state after taking the action u

    Return:
    ======
    Return an integer which represent the estimation of the value of p(x2|x1, u)

    """
    estimation = 0
    n = 0
    for i in range(0, len(trajectory) - 3, 3):
        if trajectory[i] == x1 and trajectory[i + 1] == u:
            n += 1
            if trajectory[i + 3] == x2:
                estimation += 1  # add the I function
    if n == 0:
        return 0
    else:
        return estimation / n


def state_space(trajectory):
    """
    Initialize the state space of every state that our agent has met

    Return:
    ======
    return a list of list. So a list of state.
    """
    space = []
    for i in range(0, len(trajectory), 3):
        if (trajectory[i] in space) is False:
            space.append(trajectory[i])
    return space


def sub_q_function(x, u, historic, n, X):
    """
    Computes Q(x, u) from the estimation of p(x2|x1, u) and the estimation of r(x, u) using the dynamic programming
    principle. We remind that γ is equal to 0.99 in this assignment. We remind that 'a' is the action space

    Argument:
    ========
    x : is the state
    u : is the action
    n : is the number of the iteration of the Q_function
    trajectory : is the historic of the different move, action and reward that the agent meet during the episode

    Return:
    =======
    return an integer which represent the result of the Q function i-e the state-action value function. It's an
    estimation because we estimate p(x2|x1, u) and r(x, u).
    """

    if n == 0:
        return 0
    else:
        r = estimation_r(historic, x, u)
        sum = 0
        for i in X:
            list_q = []  # we try to find the max in this list
            p = estimation_p(historic, x, u, i)
            for action in a:
                list_q.append(sub_q_function(i, action, historic, n-1, X))
            value_max = max(list_q)
            sum += p * value_max
        return r + (0.99 * sum)


def q_function(x, u, historic, n):
    """

    This final function compute approximate the Q-function in n step for the action u and the state x
    beside the historic of our agent.


    """
    X2 = state_space(historic)
    return sub_q_function(x, u, historic, n, X2)


if __name__ == '__main__':

    # ===================== test of the estimation functions =================

    historic, traj = simulation(domain, [2, 4], 200)
    """
    print(historic)
    print(domain)
    r = estimation_r(historic, [2, 4], [1, 0])
    p = estimation_p(historic, [2, 4], [1, 0], [3, 4])
    print('The reward estimate for the state [2,4] and the action [1,0] is : {} ' .format(r))
    print('The probability estimate for being in the state [3,4] from the state [2,4] by taking the action [1,0] is : \
    {}'.format(p))
    print(state_space(historic))
    print(traj)
    """

    # ======================== test of the q function ========================

    number = q_function([2, 4], [1, 0], historic, 3)
    print(number)






