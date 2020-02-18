import Section6_2
from matplotlib import pyplot as plt


if __name__ == '__main__':
    episode = range(100)

    error_1 = Section6_2.protocol_1()
    error_2 = Section6_2.protocol_1(discount_factor=0.4)

    fig, axs = plt.subplots(1, 1)
    axs.plot(episode, error_1, label='$\gamma$=0.99')
    axs.plot(episode, error_2, label='$\gamma$=0.4')
    axs.set_ylabel('|| $\hat{Q}$ - $Q$ ||$_\infty$')
    axs.set_xlabel('episode')
    axs.set_title('Convergence of $\hat{Q}$ to $Q$')
    axs.legend()
    plt.show()
