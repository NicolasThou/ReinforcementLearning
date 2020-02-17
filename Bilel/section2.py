import numpy as np
import numpy.random
from matplotlib import pyplot as plt
import time
import environment as env


def policy(x):  # random uniform policy
    i = numpy.random.randint(low=0, high=4)
    actions = list(env.action_space.keys())
    return actions[i]


def draw(position, str_x):  # display the environment
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('off')
    cell_text = np.asarray(env.rewards, dtype=np.str)
    colors = [["w" for i in range(env.m)] for j in range(env.n)]
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
        x = env.f(s, u, 0.1)  # compute new state
        r = round((0.99**t)*env.rewards[x[0]][x[1]], 4)  # compute reward
        s = x  # update state
        t += 1
        str_x = "state = " + str(x) + " | action = " + str(u) + " | reward = " + str(r) + " | t = " + str(t)
        print(str_x)
        draw(s, str_x)
        time.sleep(2)  # in order to have time to visualizes the environment
