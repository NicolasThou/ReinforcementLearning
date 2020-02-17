import numpy as np
import numpy.random
from matplotlib import pyplot as plt
import time


rewards = [[-3, 1, -5, 0, 19], [6, 3, 8, 9, 10], [5, -8, 4, 1, -8], [6, -9, 4, 19, -5], [-20, -17, -4, -3, 9]]
n = 5
m = 5

state_space = [(i, j) for i in range(n) for j in range(m)]

action_space = {
    "right": (0, 1),
    "left": (0, -1),
    "up": (-1, 0),
    "down": (1, 0)
}


def policy(x):  # random uniform policy
    i = numpy.random.randint(low=0, high=4)
    actions = list(action_space.keys())
    return actions[i]


def f(x, action, beta=0.0):  # dynamic
    rng = numpy.random.default_rng()
    w = rng.uniform()
    if w <= 1-beta:
        if action in action_space:
            i, j = action_space[action]
            new_state = (min(max(x[0] + i, 0), n-1), min(max(x[1] + j, 0), m-1))

            return new_state
        else:
            print("action not possible")
            return None, None
    else:
        return 0, 0


def draw(position, str_x):  # display the environment
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('off')
    cell_text = np.asarray(rewards, dtype=np.str)
    colors = [["w" for i in range(m)] for j in range(n)]
    # colors = [["w", "w", "w", "w", "w"], ["w", "w", "w", "w", "w"], ["w", "w", "w", "w", "w"],
    # ["w", "w", "w", "w", "w"], ["w", "w", "w", "w", "w"]]
    colors[position[0]][position[1]] = "r"
    ax.table(cellText=cell_text, cellColours=colors, cellLoc='center', loc='center',
             colWidths=[0.07, 0.07, 0.07, 0.07, 0.07])
    plt.title(str_x, fontdict={'fontsize': 8})
    plt.show()


if __name__ == '__main__':
    s = (3, 0)  # initial state
    t = 0  # time
    str_x = "state = " + str(s) + " | t = " + str(t)
    print(str_x)
    draw(s, str_x)

    while True:
        u = policy(s)  # compute the policy with actual state
        x = f(s, u, 0.1)  # compute new state
        r = round((0.99**t)*rewards[x[0]][x[1]], 4)  # compute reward
        s = x  # update state
        t += 1
        str_x = "state = " + str(x) + " | action = " + str(u) + " | reward = " + str(r) + " | t = " + str(t)
        print(str_x)
        draw(s, str_x)
        time.sleep(2)  # in order to have time to visualizes the environment
