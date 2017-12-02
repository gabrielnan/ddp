import os

import autograd.numpy as np
import matplotlib.pyplot as plt


def run_dynamics(u_old, system, dt, x_init, x_old=None, u_deltas=None):
    x_new = [x_init]
    u_new = []
    x = x_init
    for t in range(len(u_old)):
        u = u_old[t]
        if u_deltas is not None and x_old is not None:
            x_delta = system.x_delta(x, x_old[t])
            x_delta = np.concatenate([[1], x_delta])
            u = u + u_deltas[t] @ x_delta
        x = system.step(x, u, dt)
        u_new.append(u)
        x_new.append(x)
    return np.array(x_new), np.array(u_new)


def plot_costs(costs, filename='costs', path='vis/costs', file_ext='png'):
    plt.figure()
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Iteration')
    plt.savefig(os.path.join(path, filename + '.' + file_ext))
    plt.show()


