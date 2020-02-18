import numpy as np
from matplotlib import pyplot as plt
import random
from section2 import *
from section3 import *
from section4 import *

"""
Sample trajectories from a random uniform policy with a finite, large enough time horizon T. Implement a routine 
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
    for index, action_boucle in enumerate(a):  # we look for the corresponding column in the table in compared to a
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
        if trajectory[i+2] == -3:
            if state == [1,0] and action == [-1, 0]:
                r_table[index_row, index_col] = trajectory[i+2]
            elif state == [0, 1] and action == [0, -1]:
                r_table[index_row, index_col] = trajectory[i + 2]
            elif state == [0, 0] and action == [-1, 0]:
                r_table[index_row, index_col] = trajectory[i + 2]
            elif state == [0, 0] and action == [0, -1]:
                r_table[index_row, index_col] = trajectory[i + 2]
            elif state == [4, 2] and action == [0, 1]:
                r_table[index_row, index_col] = trajectory[i + 2]
            elif state == [3, 3] and action == [1, 0]:
                r_table[index_row, index_col] = trajectory[i + 2]
            elif state == [4, 4] and action == [-1, 0]:
                r_table[index_row, index_col] = trajectory[i + 2]
            elif state == [4, 3] and action == [1, 0]:
                r_table[index_row, index_col] = trajectory[i + 2]
        else:
            r_table[index_row, index_col] = trajectory[i + 2]

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


def offline_q_learning(trajectory, q_table, r_table):
    """
    Compute iteratively Q(xk, uk) with α = 0.05.

    Argument:
    =========
    trajectory : is the historic of the agent (x(t), u(t), r(t), x(t+1) ... )

    Return:
    ======
    Return the q_table where the column are : down, up, right, left
    """

    for i in range(0, len(trajectory)-3, 3):  # we look for each (x(k), u(k)) to update the q_table
        index = index_state_table(trajectory[i])  # index of x(k) in the q_table
        index2 = index_state_table(trajectory[i+3])  # index of x(k+1) in the q_table
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
        space.append(a[indice]) # we add the action corresponding in the action space "a" of the assignment
    return space


r_table = r_table_initialize()
r_table = r_table_update(r_table, experience)
q_table = q_table_initialize()
q_table = offline_q_learning(experience, q_table, r_table)
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
        index = index_state_table(new_state)  # index of x(k+1) in the q_table
        index2 = index_action_table(action)  # index of the action in the q_table
        reward = r_table[index, index2]
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


"""

The first experimental protocol is the following. The agent trains over 100 episodes
having 1000 transitions in the domain instance described in Figure 1. An episode always starts from the initial state 
(see Figure 1). The learning rate α is equal to 0.05 and the exploration rate is equal to 0.25. The values of α and 
epsilon are both constant over time. The function Qˆ is updated after every transition. The transitions are used
only once for updating Qˆ.

"""

"""========================================= section 6.2 First protocol =========================================="""

r_table2 = r_table_initialize()
q_table2 = q_table_initialize()

for i in range(100):
    historic3, all_state_met3 = simulation(domain, [3, 0], 1000)  # here we simulate a 1000 step trip from
    # an agent who moves randomly (uniform)
    r_table2 = r_table_update(r_table2, historic3)
    q_table2 = offline_q_learning(historic3, q_table2, r_table2) # here we update the q-table at each step

action_space_policy_section_6_2 = policy_from_Q(q_table2)


def final_policy_section6_2(state):
    """
    This function compute the action to do, when we put as an input the state as a epsilon-greedy policy
    with exploration rate epsilonis equal to 0.25
    Args:
        state: is the state
    Returns:
        return the action

    """

    index = index_state_table(state)
    random_number = random.uniform(0, 1)
    if random_number <= 0.25:
        return policy_random(state)
    else:
        return action_space_policy_section_6_2[index]


def value_function_section6_2(x, n):
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
        action = final_policy_section6_2(x)  # here we use the policy
        new_state = next_state(domain, x, action)
        index = index_state_table(new_state)  # index of x(k+1) in the q_table
        index2 = index_action_table(action)  # index of the action in the q_table
        reward = r_table2[index, index2]
        return reward + 0.99 * value_function_section6_2(new_state, n-1)


def display_section6_2(step):
    """
    Display J_N_µ^*(x) for each state x
    """
    x = [0, 0]
    n, m = np.shape(domain)
    tableau = []  # table of J_N_µ(x) for each state x
    for i in range(n):
        row = []
        for j in range(m):
            x[0] = i
            x[1] = j
            row.append(value_function_section6_2(x, step))  # here we add the value_function for the state x
        tableau.append(row)

    tableau = np.array(tableau)
    print(tableau)


"""========================================= section 6.2 Second protocol =========================================="""


def offline_q_learning_second_protocol(trajectory, q_table, r_table):
    """
    Compute iteratively Q(xk, uk) with α = 0.05 * 0.8 ** n

    Argument:
    =========
    trajectory : is the historic of the agent (x(t), u(t), r(t), x(t+1) ... )

    Return:
    ======
    Return the q_table where the column are : down, up, right, left
    """

    for i in range(0, len(trajectory)-3, 3):  # we look for each (x(k), u(k)) to update the q_table
        n = 0
        index = index_state_table(trajectory[i])  # index of x(k) in the q_table
        index2 = index_state_table(trajectory[i+3])  # index of x(k+1) in the q_table
        index3 = index_action_table(trajectory[i+1])  # index of the action in the q_table
        max = max_Q(index2, q_table)
        reward = r_table[index, index3]
        q_table[index, index3] = ((1-(0.05 * (0.8 ** n))) * q_table[index, index3]) + ((0.05 * (0.8 ** n)) * (reward + (0.99 * max)))
        n += 1

    return q_table


r_table3 = r_table_initialize()
q_table3 = q_table_initialize()

for i in range(100):
    historic3, all_state_met3 = simulation(domain, [3, 0], 1000)  # here we simulate a 1000 step trip from
    # an agent who moves randomly (uniform)
    r_table3 = r_table_update(r_table3, historic3)
    q_table3 = offline_q_learning_second_protocol(historic3, q_table3, r_table3)

action_space_policy_section_6_2_second_protocol = policy_from_Q(q_table3)


def final_policy_section6_2_second_protocol(state):
    """
    This function compute the action to do, when we put as an input the state as a epsilon-greedy policy
    with exploration rate epsilonis equal to 0.25
    Args:
        state: is the state
    Returns:
        return the action

    """

    index = index_state_table(state)
    random_number = random.uniform(0, 1)
    if random_number <= 0.25:
        return policy_random(state)
    else:
        return action_space_policy_section_6_2_second_protocol[index]


def value_function_section6_2_second_protocol(x, n):
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
        action = final_policy_section6_2_second_protocol(x)  # here we use the policy
        new_state = next_state(domain, x, action)
        index = index_state_table(new_state)  # index of x(k+1) in the q_table
        index2 = index_action_table(action)  # index of the action in the q_table
        reward = r_table3[index, index2]
        return reward + 0.99 * value_function_section6_2_second_protocol(new_state, n-1)


def display_section6_2_second_protocol(step):
    """
    Display J_N_µ^*(x) for each state x
    """
    x = [0, 0]
    n, m = np.shape(domain)
    tableau = []  # table of J_N_µ(x) for each state x
    for i in range(n):
        row = []
        for j in range(m):
            x[0] = i
            x[1] = j
            row.append(value_function_section6_2_second_protocol(x, step))  # here we add the value_function for the state x
        tableau.append(row)

    tableau = np.array(tableau)
    print(tableau)


"""=============================================== section 6.3 ====================================================="""


def offline_q_learning_section_6_3(trajectory, q_table, r_table):
    """
    Compute iteratively Q(xk, uk) with α = 0.05.

    Argument:
    =========
    trajectory : is the historic of the agent (x(t), u(t), r(t), x(t+1) ... )

    Return:
    ======
    Return the q_table where the column are : down, up, right, left
    """

    for i in range(0, len(trajectory)-3, 3):  # we look for each (x(k), u(k)) to update the q_table
        index = index_state_table(trajectory[i])  # index of x(k) in the q_table
        index2 = index_state_table(trajectory[i+3])  # index of x(k+1) in the q_table
        index3 = index_action_table(trajectory[i+1])  # index of the action in the q_table
        max = max_Q(index2, q_table)
        reward = r_table[index, index3]
        q_table[index, index3] = ((1-0.05) * q_table[index, index3]) + (0.05 * (reward + (0.4 * max)))

    return q_table

r_table4 = r_table_initialize()
q_table4 = q_table_initialize()

for i in range(100):
    historic3, all_state_met3 = simulation(domain, [3, 0], 1000)  # here we simulate a 1000 step trip from
    # an agent who moves randomly (uniform)
    r_table4 = r_table_update(r_table4, historic3)
    q_table4 = offline_q_learning(historic3, q_table4, r_table4)

action_space_policy_section_6_3 = policy_from_Q(q_table4)


def final_policy_section6_3(state):
    """
    This function compute the action to do, when we put as an input the state as a epsilon-greedy policy
    with exploration rate epsilonis equal to 0.25
    Args:
        state: is the state
    Returns:
        return the action

    """

    index = index_state_table(state)
    random_number = random.uniform(0, 1)
    if random_number <= 0.25:
        return policy_random(state)
    else:
        return action_space_policy_section_6_3[index]


def value_function_section6_3(x, n):
    """
    Compute the return of a stationary policy thanks to the Bellman Equation
    We use the policy that we found to compute each J(x)
    The discount factor gamma is equal to 0.4

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
        action = final_policy_section6_3(x)  # here we use the policy
        new_state = next_state(domain, x, action)
        index = index_state_table(new_state)  # index of x(k+1) in the q_table
        index2 = index_action_table(action)  # index of the action in the q_table
        reward = r_table4[index, index2]
        return reward + 0.4 * value_function_section6_3(new_state, n-1)


def display_section6_3(step):
    """
    Display J_N_µ^*(x) for each state x
    """
    x = [0, 0]
    n, m = np.shape(domain)
    tableau = []  # table of J_N_µ(x) for each state x
    for i in range(n):
        row = []
        for j in range(m):
            x[0] = i
            x[1] = j
            row.append(value_function_section6_3(x, step))  # here we add the value_function for the state x
        tableau.append(row)

    tableau = np.array(tableau)
    print(tableau)



if __name__ == '__main__':

    """============================================ section 6.1 =============================================="""

    print('---------here the domain/environment---------')
    print(domain)
    print('---------here the experience of the agent---------')
    print(experience)
    print('---------here the r_table---------')
    print(r_table)
    print('--------- here the q_table ---------')
    np.set_printoptions(suppress=True)
    print(q_table)
    print('--------- here the action space for the policy ---------')
    print(action_space_policy)
    print('--------- test of the policy and its return ---------')
    print(final_policy_section6([3,4]))
    print('-------- J_N_µ^*(x) for each state x ---------')
    display_section6(500)
    print()
    print('-------- J_N_µ*(x) for each state x ---------')
    display_section4(500)

    print("============================================ section 6.2 ==============================================")
    print("============================================ First Protocol ===========================================")

    print()
    print('---------here the r_table2---------')
    print(r_table2)
    print('--------- here the q_table2 ---------')
    np.set_printoptions(suppress=True)
    print(q_table2)
    print('--------- here the action space for the greedy policy ---------')
    print(action_space_policy_section_6_2)
    print('--------- test of the policy6_2 and its return ---------')
    print(final_policy_section6_2([3, 4]))
    print('-------- J_N_µ^*(x) for each state x ---------')
    display_section6_2(500)

    print("============================================ Second Protocol ===========================================")

    print()
    print('---------here the r_table3---------')
    print(r_table3)
    print('--------- here the q_table3 ---------')
    np.set_printoptions(suppress=True)
    print(q_table3)
    print('--------- here the action space for the greedy policy with different learning rate ---------')
    print(action_space_policy_section_6_2_second_protocol)
    print('--------- test of the policy6_2_second_protocol and its return ---------')
    print(final_policy_section6_2_second_protocol([3, 4]))
    print('-------- J_N_µ^*(x) for each state x ---------')
    display_section6_2_second_protocol(500)

    """
    Here the learning rate decrease, and we can notice that we don't have a better result from a learning rate constant
    """
    print()
    print("============================================ Section 6.3 ===========================================")
    print()
    print('---------here the r_table4---------')
    print(r_table4)
    print('--------- here the q_table4 ---------')
    np.set_printoptions(suppress=True)
    print(q_table4)
    print('--------- here the action space for a discount factor equal to 0.4 ---------')
    print(action_space_policy_section_6_3)
    print('--------- test of the policy6_3 and its return ---------')
    print(final_policy_section6_3([3, 4]))
    print('-------- J_N_µ^*(x) for each state x ---------')
    display_section6_3(500)












