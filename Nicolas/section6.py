import numpy as np
from matplotlib import pyplot as plt
import random
from Nicolas.section2 import *
from Nicolas.section3 import *
from Nicolas.section4 import *

"""
1 - Sample trajectories from a random uniform policy with a finite, large enough time horizon T. Implement a routine 
which iteratively updates Qb(xk , uk ) with Q-learning from these trajectories. Run your routine with a constant 
learning rate α = 0.05. Derive directly µb∗ from Qb. Display JN µb∗ , along with JN µ∗ , for each initial state x.

"""

def q_table_initialize():
    """
    initialize the q table at 0 for everywhere, where there is
    n column for each action and m row for each state
    """
    return np.zeros((25, 4))

def r_table_initialize():
    """
    initialize the reward table at 0 for everywhere, where there is
    n column for each action and m row for each state
    """
    return np.zeros((25, 4))


def index_state_table(state):
    """
    This function compute the index of the row in the q_table with a state for input
    Args:
        state: the state which correspond to the row in the table

    Returns:
        return an index of the row in the table

    """
    return state[0]*5 + state[1]

def state_index_table(index):
    """
    This function compute the state of the row in the table with a index for input
    Args:
        index: the index which correspond to the row in the table

    Returns:
        return a state of the corresponding row in the table

    """
    i = index // 5
    j = index - i
    return [i, j]

def index_action_table(action):
    """
    This function compute the index of the column in the table with a action for input
    Args:
        action: the action which correspond to the column in the table

    Returns:
        return an index of the column in the table

    """
    for index, action_boucle in enumerate(a):
        if action_boucle == action:
            return index


def r_table_update(r_table, trajectory):
    """
    This function update the reward for each couple (x,u) that the agent met

    Args:
        r_table: is the reward table
        trajectory: is the historic of the interaction with the environment that the agent has made

    Returns:
        return the r_table
    """
    for i in range(0, len(trajectory)-3, 3):
        state = trajectory[i]
        action = trajectory[i+1]
        index_row = index_state_table(state)
        index_col = index_action_table(action)
        r_table[index_row, index_col] = trajectory[i+2]

    return r_table


def max_Q(index, q_table):
    """
    This function return the max of Q(index, u) for every u action available

    Args:
    ====
        index: is the index of the state in the q_table where we try to find the best Q-value for applying each action
        q_table : is the q_table

    Returns:
    =======
        return a float which is the q-value

    """

    return np.max(q_table[index])


def offline_q_learning(q_table, r_table, n):
    """
    Compute iteratively Q(xk, uk) with α = 0.05.

    Argument:
    =========
    trajectory : is the historic of the agent (x(t), u(t), r(t), x(t+1) )

    Return:
    ======
    Return a value
    """

    for k in range(n):
        for i in range(25):  # for each state
            for j in range(4):  # for each action
                reward = r_table[i, j]
                max = max_Q(i, q_table)
                q_table[i, j] = ((1-0.05) * q_table[i, j]) + (0.05 * (reward + (0.99 * max)))

    return q_table


if __name__ == '__main__':

    r_table = r_table_initialize()
    r_table = r_table_update(r_table, experience)
    q_table = q_table_initialize()
    q_table = offline_q_learning(q_table, r_table, 50)

    print(domain)
    print('here the experience of the agent')
    print(r_table)
    print(q_table)








