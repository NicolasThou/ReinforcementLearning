import numpy as np


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


def f(x, action, beta=0.0):  # dynamic
    rng = np.random.default_rng()
    w = rng.uniform()
    if w <= 1-beta:
        if action in action_space:
            i, j = action_space[action]

            return min(max(x[0] + i, 0), n-1), min(max(x[1] + j, 0), m-1)
        else:
            print("action not possible")
            return None, None
    else:
        return 0, 0
