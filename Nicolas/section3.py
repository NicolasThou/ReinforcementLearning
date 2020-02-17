import numpy as np
from matplotlib import pyplot as plt
import random
from Nicolas.section2 import *

"""

Implement a routine to estimate J^µ in this domain, where µ : X → U is a stationary
policy. The implementation should exploit the dynamic programming principle where
a sequence of functions J_N for N = 0, 1, 2... is computed using the Bellman Equation.
Test your implementation with your rule-based policy of Section 2. Choose a N which
is large enough to approximate well the infinite time horizon of the policy and motivate
your choice. Display J_N_µ(x) for each state x.

"""


def value_function(x, n):
    """
    Compute the return of a stationary policy thanks to the Bellman Equation
    We use the policy_random to compute each J(x)
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
        action = policy_random(x)  # here we use the policy
        new_state = next_state(domain, x, action) # here we have disturbance, with certain probability
        reward = domain[new_state[0], new_state[1]]
        return reward + 0.99 * value_function(new_state, n-1)


def display2(step):
    """
    Display J_N_µ(x) for each state x
    """
    print(domain)
    x = [0, 0]
    n, m = np.shape(domain)
    tableau = []  # table of J_N_µ(x) for each state x
    for i in range(n):
        row = []
        for j in range(m):
            x[0] = i
            x[1] = j
            row.append(value_function(x, step))  # here we add the value_function for the state x
        tableau.append(row)

    tableau = np.array(tableau)
    print(tableau)


if __name__ == '__main__':
    display2(600)  # we can choose a N which is large enough to approximate well the infinite time horizon of the policy



