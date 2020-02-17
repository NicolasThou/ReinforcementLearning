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

"""============================================ section 6.1 =============================================="""


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


def offline_q_learning(trajectory, q_table, r_table, n):
    """
    Compute iteratively Q(xk, uk) with α = 0.05.

    Argument:
    =========
    trajectory : is the historic of the agent (x(t), u(t), r(t), x(t+1) )

    Return:
    ======
    Return a value
    """

    for i in range(0, len(trajectory)-3, 3):
        index = index_state_table(trajectory[i])  # index of actual state in the q_table
        index2 = index_state_table(trajectory[i+3])  # index of next state in the q_table
        index3 = index_action_table(trajectory[i+1])  # index of the action in the q_table
        max = max_Q(index2, q_table)
        reward = r_table[index, index3]
        q_table[index, index3] = ((1-0.05) * q_table[index, index3]) + (0.05 * (reward + (0.99 * max)))

    return q_table

def policy_from_Q(q_table):
    """
    This function compute the policy
    Args:
        q_table: is the q_table for each action and each state we have the q_value

    Returns:
        return the action space

    """
    size = np.shape(q_table)
    space = []
    for i in range(size[0]):
        indice = np.argmax(q_table[i])
        space.append(a[indice])
    return space


r_table = r_table_initialize()
r_table = r_table_update(r_table, experience)
q_table = q_table_initialize()
q_table = offline_q_learning(experience, q_table, r_table, 50)
action_space_policy = policy_from_Q(q_table)

def final_policy_section6(state):
    """
    This function compute the action to do, when we put as an input the state
    Args:
        state: is the state
    Returns:
        return the action

    """
    index = index_state_table(state)
    return action_space_policy[index]


def value_function_section6(x, n):
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
        action = final_policy_section6(x)  # here we use the policy
        new_state = next_state(domain, x, action)
        reward = domain[new_state[0], new_state[1]]
        return reward + 0.99 * value_function_section6(new_state, n-1)


def display_section6(step):

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
            row.append(value_function_section6(x, step))  # here we add the value_function for the state x
        tableau.append(row)

    tableau = np.array(tableau)
    print(tableau)


"""============================================ section 6.2 =============================================="""


r_table = r_table_initialize()
r_table = r_table_update(r_table, experience)
q_table = q_table_initialize()
q_table = offline_q_learning(experience, q_table, r_table, 50)
action_space_policy = policy_from_Q(q_table)

for i in range(100):
    historic3, all_state_met3 = simulation(domain, [3, 0], 10000)  # here we simulate a 5000 step trip from an agent who moves randomly (uniform), usefull for the section4




if __name__ == '__main__':

    """============================================ section 6.1 =============================================="""

    print('---------here the domain/environment---------')
    print(domain)
    print('---------here the experience of the agent---------')
    print(experience)
    print('---------here the r_table---------')
    print(r_table)
    print('--------- here the q_table ---------')
    print(q_table)
    print('--------- here the action space for the policy ---------')
    print(action_space_policy)
    print('--------- test of the policy and its return ---------')
    print(final_policy_section6([3,4]))
    print('-------- J_N_µ(x) for each state x ---------')
    display_section6(500)

    """============================================ section 6.2 =============================================="""









