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
        r = estimation_r(h, x, u)  # We estimate the reward according the x and u we've ever met
        sum = 0
        X = state_space(x)  # We update state space that it's possible to reach from x
        for i in X:
            list_q = []  # we try to find the max in this list
            p = estimation_p(h, x, u, i)
            for action in a:
                list_q.append(q_function(i, action, h, n-1))
            value_max = max(list_q)
            sum += p * value_max
        return r + (0.99 * sum)


def compute_policy(h, n):
    """
    This function estimate for each state met in the experience, the best action to do.

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
        act.append(np.argmax(compare))

    return trajectory, act


def final_policy(x, trajectory, act):
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
    for i, j in enumerate(trajectory):
        if j == x:
            return a[act[i]]
    print('the state in parameter has never been meet yet')
    return a[0]  # random move



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
    print('All the state that the agent has met, are : ')
    state_met = state_space2(experience)  # display of each state that the agent met
    np.save('./npy/state_met', state_met)
    print(state_met)
    state_met2 = state_space2(experience2)  # display of each state that the agent met
    np.save('./npy/state_met2', state_met2)
    print(state_met2)

    # ===================== test of the estimation functions =================

    """

    print(experience)
    r = estimation_r(experience, [2, 4], [1, 0])
    p = estimation_p(experience, [2, 4], [1, 0], [3, 4])
    print('The reward estimate for the state [2,4] and the action [1,0] is : {} ' .format(r))
    print('The probability estimate for being in the state [3,4] from the state [2,4] by taking the action [1,0] is : \
    {}'.format(p))
    print(state_met)

    """

    # ======================== test of the q function ========================

    # We want to approximate the Q-function for a state [3,0] and the move
    # to down


    number = q_function([3, 0], [1, 0], experience, 3)
    print('from the state [3,0] and the action [1 , 0], the Q-function value has a value of :')
    print(number)



    # ======================= test of the computing policy ===================
    """

    t, act = compute_policy(experience, 3)  # t is like state_met
    t2, act2 = compute_policy(experience2, 3)  # t2 is like state_met2

    np.save('./npy/policy1', act)
    np.save('./npy/policy2', act2)

    print()
    print('here is the action space for the first agent')  # print all the state that the agent has met
    print(act)  # print the best action corresponding at each state that the agent has met

    print('here is the action space for the second agent')  # print all the state that the agent has met
    print(act2)  # print the best action corresponding at each state that the agent has met

    """

    # ======================= test of the estimation of the best policy ===================

    """
    action = final_policy([2,4], t, act)
    print('the best action to do from the state [2,4] is')
    print(action)
    """

    # ======================= Display J_N_µ∗ for each state x ===================

    # Comment all the test before testing this test, even the 8 first line of the main

    """
    s1 = np.load('./npy/state_met.npy', allow_pickle=True)
    s1 = s1.tolist()
    s2 = np.load('./npy/state_met2.npy', allow_pickle=True)
    s2 = s2.tolist()
    pol1 = np.load('./npy/policy1.npy', allow_pickle=True)
    pol1 = pol1.tolist()
    pol2 = np.load('./npy/policy2.npy', allow_pickle=True)
    pol2 = pol2.tolist()

    print(s1)
    print(pol1)
    print(s2)
    print(pol2)

    print(value_function([0, 3], s1, pol1, 500))
    print(value_function([0, 3], s2, pol2, 500))

    """






