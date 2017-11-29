import autograd.numpy as np
from autograd import grad
from autograd import jacobian
import sys
import matplotlib.pyplot as plt


def forward_pass(x_init, x_old, u_bar, u_deltas, F, T, dt):
    grad()
    x_bar = [x_init]
    x = x_init
    for t in range(T):
        u_delta = u_deltas[t]
        x_delta = x - x_old[t]
        x += F(x, u_bar + u_delta(x_delta)) * dt
        x[2] %= 2 * np.pi
        x_bar.append(x)
    return x_bar


def plot_costs(costs):
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Iteration')
    plt.show()


def get_lagrangians(lagr, dt):
    L = lambda x: lagr(x) * dt
    Lx = grad(L, 0)
    Lu = grad(L, 1)
    Lxx = jacobian(Lx, 0)
    Lxu = jacobian(Lx, 1)
    Luu = jacobian(Lu, 1)

    return L, Lx, Lu, Lxx, Lxu, Luu


def ddp(x_init, n, m, phi, lagr, F, dt, T=1000, max_iters=100, epsilon=0.0,
        u_bar=None):
    """Runs Differential Dynamic Programming (DDP)

    :param x_init: initial state (np.array)
    :param n: dimension of state
    :param m: dimension of control
    :param phi: terminal cost function
    :param lagr: running cost function
    :param F: dynamics function F(state, control)
    :param dt: time step
    :param T: time horizon
    :param max_iters: maximum number of iterations
    :param epsilon: cost error to terminate algorithm
    :param u_bar: initial nominal control sequence (if None, random sequence
                  will be used)
    :return: tuple<optimal control sequence, list<costs of one iteration>>
    """

    # Initialize variables
    if len(x_init) != n:
        raise ValueError('initial state does not agree with state dimension n')
    u_bar = np.random.rand(T, m) if u_bar is None else u_bar
    x_bar = None
    u_deltas = None

    # Differentiation
    L, Lx, Lu, Lxx, Lxu, Luu = get_lagrangians(lagr, dt)
    Fx = grad(F, 0)
    Fu = grad(F, 1)

    # Dynamics Linearization and Discretization
    Phi = lambda x: np.eye(n) + Fx(x) * dt
    Beta = lambda x: Fu(x) * dt

    iter = 0
    cost = sys.maxsize
    costs = []
    while cost > epsilon and iter < max_iters:
        x_bar = forward_pass(x_init, x_bar, u_bar, u_deltas, F, T, dt)
        u_deltas = []

        x_final = x_bar[T]
        V = phi(x_final)
        Vx = grad(phi)(x_final)
        Vxx = jacobian(phi)(x_final)

        cost = V

        for t in range(T - 1, -1, -1):
            # Setup
            x = x_bar[t]
            Phi_x = Phi(x)
            Beta_x = Beta(x)
            cost += L(x)

            # Compute Q's
            Q = L(x) + V
            Qx = Lx(x) + np.dot(Phi_x.T, Vx)
            Qu = Lu(x) + np.dot(Beta_x.T, Vx)
            Qxx = Lxx(x) + np.dot(Phi_x.T, Vxx, Phi_x)
            Qxu = Lxu(x) + np.dot(Phi_x.T, Vxx, Beta_x)
            Quu = Luu(x) + np.dot(Beta_x.T, Vxx, Beta_x)

            # Compute optimal change
            Quu_inv = np.linalg.inv(Quu)
            u_delta = lambda x_delta: - np.dot(Quu_inv,
                                               Qu + np.dot(Qxu.T, x_delta))
            u_deltas.append(u_delta)

            # Compute V's at time t
            V = Q - 0.5 * np.dot(Qu[np.newaxis].T, Quu_inv, Qu)
            Vx = Qx - np.dot(Qxu, Quu_inv, Qu)
            Vxx = Qxx - np.dot(Qxu, Quu_inv, Qxu.T)

        iter += 1
        costs.append(cost)
        # x_finals.append(x_bar[T])

    # Plot costs
    plot_costs(costs)

    return u_bar, costs
