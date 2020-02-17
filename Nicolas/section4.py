import numpy as np
from matplotlib import pyplot as plt
import random
from Nicolas.section3 import *
from Nicolas.section2 import *

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
    Computes Q(x, u) from the estimation of p(x2|x1, u) and the estimation of r(x, u) using the dynamic programming
    principle. We remind that γ is equal to 0.99 in this assignment. We remind that 'a' is the action space

    Argument:
    ========
    x : is the state
    u : is the action
    n : is the number of the iteration of the Q_function
    h : is the experienceof the different move, action and reward that the agent meet during the episode

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


def compare_policy(h1, action_space1, h2, action_space2):
    """
    compare two policy
    
    Argument:
    ========
    h1 and h2 : are the two experience of the 2 agents
    action_space1 and action_space2 : are the two space actions corresponding to h1 and h2
    
    Return:
    ======
    Return a purcentage of likeness
    
    """
    n = 0
    for index1, i in enumerate(h1):
        for index2, j in enumerate(h2):
            if i == j:
                if action_space1[index1] == action_space2[index2]:
                    n += 1
    total = min(len(h1), len(h2))
    return (n/total) * 100


def value_function(x, s, act, n):
    """
    Compute the return of a policy thanks to the Bellman Equation
    We use the policy that we found to compute each J(x)
    The discount factor gamma is equal to 0.99

    Argument:
    ========
    x : is the initial state
    n : number of value function
    s : is every state that the agent know
    act : are the actions that the agent take for each state corresponding in s

    Return:
    =======
    return an integer, which is the estimation of J^µ

    """
    if n == 0:
        return 0
    else:
        for index, i in enumerate(s):
            if x == i:
                action = a[act[index]]  # a is the action space, act[index] is the index of the space action to take
                new_state = next_state(domain, x, action)
                reward = domain[new_state[0], new_state[1]]
                return reward + 0.99 * value_function(new_state, s, act, n-1)
        return 0


if __name__ == '__main__':


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


    #We want to approximate the Q-function for a state [3,0] and the move
    #to down

    """
    number = q_function([3, 0], [1, 0], experience, 4)
    print('from the state [3,0] and the action [1 , 0], the Q-function value has a value of :')
    print(number)
    
    """

    # ======================= test of the computing policy ===================

    t, act = compute_policy(experience, 3)  # t is like state_met
    t2, act2 = compute_policy(experience2, 3)  # t2 is like state_met2

    np.save('./npy/policy1', act)
    np.save('./npy/policy2', act2)

    print()
    print('here is the action space for the first agent')  # print all the state that the agent has met
    print(act)  # print the best action corresponding at each state that the agent has met

    print('here is the action space for the second agent')  # print all the state that the agent has met
    print(act2)  # print the best action corresponding at each state that the agent has met

    print()
    print('The % of likeness of 2 different agent from 2 experience : the one with 1000 step and the other '\
        'with 3000 step is :')
    print(compare_policy(state_met, act, state_met2, act2))


    # ======================= test of the estimation of the best policy ===================

    """
    action = final_policy([2,4], t, act)
    print('the best action to do from the state [2,4] is')
    print(action)
    """

    # ======================= Display J_N_µ∗ for each state x ===================

    #Comment all the test before testing this test, even the 8 first line of the main

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





