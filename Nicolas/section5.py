import numpy as np
from matplotlib import pyplot as plt
import random
from Nicolas.section4 import *
from Nicolas.section3 import *
from Nicolas.section2 import *

experience = np.load('./npy/historic.npy', allow_pickle=True)
experience2 = np.load('./npy/historic2.npy', allow_pickle=True)

"""
Implement a routine which estimates r(x, u) and p(x 0|x, u) from a given trajectory
ht = (x0, u0, r0, x1, u1, r1, ... , ut−1, rt−1, xt). Display the convergence speed towards p and r.
Compute Qb by using br(x, u) and bp(x0 |x, u) that estimate the MDP structure. Derive µb∗ from Qb. Display JN
µb∗ , along with JNµ∗ , for each state x.

Run your implementation over several trajectories, generated from a random uniform
policy, of different lengths. Explain the influence of the length on the quality of the
approximation Qˆ using an infinite norm.

"""


def estimation_r(trajectory, x, u):
    """
    This function compute from a random uniform policy an estimation of r(x, u) from a trajectory of size T.
    The trajectory is like : (x(0), u(0), r(0), ..., u(t-1), r(t-1), x(t))

    Argument:
    ========
    trajectory : is the history of each action, move and reward of the agent
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
    if n == 0:  # means that we have never meet this state x and this action u together
        return -5000
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
    if n == 0:  # means that we have never meet this state x1 and this action u together
        return 0
    else:
        return estimation / n


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


def q_function(x, u, h, n):
    """
    Computes q(x, u) from the estimation of p(x2|x1, u) and the estimation of r(x, u) using the dynamic programming
    principle. We remind that γ is equal to 0.99 in this assignment. We remind that 'a' is the action space

    Argument:
    ========
    x : is the state
    u : is the action
    n : is the number of the iteration of the Q_function
    h : is the experience of the different move, action and reward that the agent meet during the episode

    Return:
    =======
    return an integer which represent the result of the Q function i-e the state-action value function. It's an
    estimation because we estimate p(x2|x1, u) and r(x, u).
    """

    if n == 0:
        return 0
    else:
        r = estimation_r(h, x, u)  # We estimate the reward according the x and u we've ever met, otherwise it's -5000
        sum = 0
        X = state_space(x)  # We update state space that it's possible to reach from x
        for i in X:
            list_q = []  # we try to find the max in this list
            p = estimation_p(h, x, u, i)
            for action in a:
                list_q.append(q_function(i, action, h, n-1))
            value_max = max(list_q)  # here we have the max, but if i and action is a pair of (x,u)
            # that the agent has never met, so the value_max will be negative
            sum += p * value_max  # So it may multiply a probabilité between 0 and 1 to a negative number
        return r + (0.99 * sum)  # We understand now that the q_value can be a big negative number


def compute_policy(h, n):
    """
    This function estimate for each state met in the experience, the best action to do.
    For each state that the agent has met, we compute the best q_value for each action.
    Then we don't add the action but the index in the action space " a " of the better action to do
    for this state. We iterate this for all the state that the agent has met.

    Argument:
    ========

    n : is the number of the iteration of the Q_function
    h : is the history of the different move, action and reward that the agent meet during the episode

    Return:
    =======
    return two list, the first one is the list of state that the agent know (input of the policy),
    the second is the corresponding best action for each state (output corresponding to the input of the policy)

    """
    trajectory = state_space2(h)  # same as the state_met
    act = []  # best action to do for each state we've met
    for state in trajectory:
        compare = []
        for action in a:
            compare.append(q_function(state, action, h, n))
        compare = np.array(compare)
        act.append(np.argmax(compare)) # argument is the index which correspond to the index in the action space a

    return trajectory, act


every_state_met, act = compute_policy(experience, 2)


def final_policy_section5(x):
    """
    return the action to do according to a state.

    Argument:
    ========
    x is a list of the coordinate
    trajectory : list of state that the agent know same as the state_met
    act : list of action corresponding to the best action to do in compare to the states of trajectory

    Return:
    =======
    return an action
    """

    for i, j in enumerate(every_state_met):
        if j == x:
            return a[act[i]]
        else:
            # the state in parameter has never been meet yet
            return a[0]  # random move


def value_function_section5(x, n):
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
        action = final_policy_section5(x)  # here we use the policy
        new_state = next_state(domain, x, action)
        reward = estimation_r(experience, new_state, action)  # here we can have a big negative number
        # if the agent has never met the pair (new_state, action)
        return reward + 0.99 * value_function_section5(new_state, n-1)


def display_section5(step):
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
            row.append(value_function_section5(x, step))  # here we add the value_function for the state x
        tableau.append(row)

    tableau = np.array(tableau)
    print(tableau)


if __name__ == '__main__':

    """
    We notice that we went a little too far in the section4. In fact, we read the course for this section5 
    during coding the section4, and we applied method for the estimation of p(x2|x1, u) and r(x, u) for the section4.
    We though that the assumption that we don't know r(x, u) neither p(x2|x1, u) was true for the section4.
    Whereas, in the section4 it was enough to just implement a little function for r which return the reward. 
    And implement also a simple function for p(x2|x1, u) which return the probability to be in x2 from x1 and taking the
    action u. Anyway, we have already done the section 5. So it's just the same functions from the section4.
    
    """

    print(domain)  # domain on what we work
    state_met = state_space2(experience)  # display of each state that the agent met
    np.save('./npy/state_met', state_met)

    print('===================== test of the estimation functions =================')

    print('here is the exploration of the agent with his environment')
    print(experience)
    r = estimation_r(experience, [2, 4], [1, 0])
    p = estimation_p(experience, [2, 4], [1, 0], [3, 4])
    print()
    print('The reward estimate for the state [2,4] and the action [1,0] is : {} ' .format(r))
    print()
    print('The probability estimate for being in the state [3,4] from the state [2,4] by taking the action [1,0] is : \
    {}'.format(p))
    print()
    print('here are all the state that the agent has met, like every_state_met')
    print(state_met)


    print('======================== test of the q function ========================')

    # We want to approximate the Q-function for a state [3,0] and the move
    # to down

    number = q_function([3, 0], [1, 0], experience, 3)
    print('from the state [3,0] and the action [1 , 0], the Q-function value has a value of :')
    print(number)


    print('======================= test of the computing policy ===================')


    print("here are all the state that agent has met")
    print(every_state_met)
    print()
    print('here is the action space for the agent')  # print all the state that the agent has met
    print(act)  # print the best action corresponding at each state that the agent has met


    print('======================= test of the estimation of the best policy ===================')

    action = final_policy_section5([2,4])
    print('the best action to do from the state [2,4] is')
    print(action)


    print(' ======================= Display J_N_µ∗ for each state x =================== ')

    display_section5(500)








