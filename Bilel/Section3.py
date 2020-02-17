import Bilel.section2 as S2


def R(x, u):  # return rewards obtained after taking action 'u' while in state 'x'
    new_state = S2.f(x, u)
    return S2.rewards[new_state[0]][new_state[1]]


def J_N(x, mu, N, discount_factor=0.99):  # computes J(mu,N,x)
    if N < 0:
        print("N cannot be negative !")
        exit()
    elif N == 0:
        return 0
    else:
        return R(x, mu(x)) + discount_factor*J_N(S2.f(x, mu(x)), mu, R, N-1)


if __name__ == '__main__':
    N = 299
    for x in S2.state_space:
        j = round(J_N(x, S2.policy, R, N), 4)
        print("x =", x, " | N =", N, " | J(N,x) =", j)
        str_x = "state : " + str(x) + " | reward : " + str(j) + " | N : " + str(N)
        S2.draw(x, str_x)
