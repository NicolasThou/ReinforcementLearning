import numpy as np
from matplotlib import pyplot as plt
import random

"""
Implement the different components of the domain. Implement a rule-based policy of
your choice (e.g., always go left, select actions always at random...). Simulate the
policy in the domain and display the trajectory.
"""

def initialize_domain():
    """
    Initialize a domain with 5 row and 5 column, for each cells there is a reward value.

    Return:
    =======
    return an array
    """

    domain = [[-3, 1, -5, 0, 19], [6, 3, 8, 9, 10], [5, -8, 4, 1, -8], [6, -9, 4, 19, -5], [-20, -17, -4, -3, 9]]
    return np.array(domain)


def action_space():
    """
    Initialize the action space

    Return:
    ======
    return a list with the 4 movement authorize
    """
    return [[1,0], [-1,0], [0,1], [0,-1]]  # down, up, right, left


domain = initialize_domain()  # domain use for the all assignement
np.save('./npy/domain', domain)
a = action_space()  # a is the action space
np.save('./npy/action_space', a)


def policy_left(x):
    """
    return the action to do according to a state. It always turning left.

    Argument:
    ========
    x is a list of the coordinate

    Return:
    =======
    return a list in action space which refers to an action (here is left)
    """

    return a[3]

def policy_random(x):
    """
    return the action to do according to a state. It choose a random action
    thanks to a random integers from the “discrete uniform” distribution

    Argument:
    ========
    x is a list of the coordinate

    Return:
    =======
    return a list in action space which refers to a random action
    """
    n = random.randint(0,len(a)-1)
    return a[n]

def next_state(domain, state, action):
    """
    Compute the next state

    Argument:
    ========
    domain : is the domain
    state : is the state where we from
    action : is the action we take to move

    return:
    ======
    return a list, which represent the new state after the move
    """

    w = random.uniform(0, 1)
    if w <= 0.5:
        size = np.shape(domain)
        n, m = size[0], size[1]
        x = min(max(state[0] + action[0], 0), n - 1)
        y = min(max(state[1] + action[1], 0), m - 1)
        new_state = [x,y]
    else:
        new_state = [0, 0]

    return new_state

def simulation(domain, initial_state, step):
    """
    Simulate the policy random in the domain

    Argument:
    ========
    domain : is the domain
    initial_state : is the state where we begin, it is a list of 2 integers [x, y]
    step : is the number of step the agent make a move

    Return:
    ======
    return the historic, and the trajectory. Here the historic is like (x(0), u(0), r(0), ..., u(t-1), r(t-1), x(t))
    and the trajectory is like (x(0), x(1), x(2), ..., x(t-1), x(t)). So in the historic we have the reward and
    the action also.
    """

    historic = [initial_state]
    trajectory = [initial_state]
    m = []
    state = initial_state

    for i in range(step):
        action = policy_random(state)  # here we use the random policy but we can change it by the left policy
        new_state = next_state(domain, state, action)
        reward = domain[new_state[0], new_state[1]]
        historic.append(action)
        historic.append(reward)
        historic.append(new_state)
        trajectory.append(new_state)
        state = new_state

    return historic, trajectory


historic, all_state_met = simulation(domain, [3, 0], 5000)  # here we simulate a 100 step trip from an agent who moves randomly (uniform), usefull for the section4
np.save('./npy/historic', historic)
historic2, all_state_met2 = simulation(domain, [3, 0], 5000)  # here we simulate a 100 step trip from an agent who moves randomly (uniform), usefull for the section4
np.save('./npy/historic2', historic2)

def simulation2(domain, initial_state, step):
    """
    Simulate the policy random in the domain and display the trajectory.

    Argument:
    ========
    domain : is the domain
    initial_state : is the state where we begin, it is a list of 2 integers [x, y]
    step : is the number of step the agent make a move

    Return:
    ======
    return the historic, and the trajectory. Here the historic is like (x(0), u(0), r(0), ..., u(t-1), r(t-1), x(t))
    and the trajectory is like (x(0), x(1), x(2), ..., x(t-1), x(t)). So in the historic we have the reward and
    the action also.
    """

    state = initial_state
    display(5, 5, domain, state)
    for i in range(step):
        action = policy_random(state)  # here we use the random policy but we can change it by the left policy
        new_state = next_state(domain, state, action)
        display(5, 5, domain, new_state)
        state = new_state


def display(n, m, domain, position):

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('off')
    cell_text = np.asarray(domain, dtype=np.str)
    colors = [["w" for i in range(m)] for j in range(n)]
    colors[position[0]][position[1]] = "r"
    ax.table(cellText=cell_text, cellColours=colors, cellLoc='center', loc='center',
             colWidths=[0.07 for i in range(m)])
    plt.show()


if __name__ == '__main__':

    #  test of the initialize_domain, and the 2 policies

    print(domain)
    a1 = policy_left([2,3])
    print(a1)  # always [1,0]
    a2 = policy_random([2,3])
    print(a2)  # random action

    #  test the simulation

    simulation2(domain, [3,0], 1000)



