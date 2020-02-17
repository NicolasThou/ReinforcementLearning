import Bilel.Section2 as S2
import environment as env


def J_N(x, mu, N, discount_factor=0.99):  # computes J(mu,N,x)
    if N < 0:
        print("N cannot be negative !")
        exit()
    elif N == 0:
        return 0
    else:
        new_state = env.f(x, mu(x))
        r = env.rewards[new_state[0]][new_state[1]]
        return r + discount_factor*J_N(env.f(x, mu(x)), mu, N-1)


if __name__ == '__main__':
    N = 299
    for x in env.state_space:
        j = round(J_N(x, S2.policy, N), 4)
        print("x =", x, " | N =", N, " | J(N,x) =", j)
        str_x = "state : " + str(x) + " | reward : " + str(j) + " | N : " + str(N)
        S2.draw(x, str_x)
