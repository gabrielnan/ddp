import autograd.numpy as np
from ddp import ddp, plot_costs


def main():
    # System Constants
    g = -9.8  # gravitational acceleration
    m1 = 0.5  # mass of car
    m2 = 0.2  # mass of pole tip
    l = 3  # pole length

    # DDP var init
    x_init = np.array([.0, .0, .0, .0])
    x_final = np.array([.0, .0, np.pi, .0])
    dt = .001  # time step
    m = 1  # control dim
    n = 4  # state dim

    # Cost vars
    Q = np.diag([1, 5, 5, 3])
    R = np.array([[1]])
    terminal_scale = 2

    # Define functions
    def F(x, u):
        pos_vel = x[1]
        theta = x[2]
        theta_vel = x[3]

        pos_acc = (-m2 * g * np.cos(theta) * np.sin(theta) + u \
                   + m2 * theta_vel ** 2 * np.sin(theta)) \
                  / (m1 + m2 - m2 * np.cos(theta) ** 2)
        theta_acc = (g * np.sin(theta) - np.cos(theta) * pos_acc) / l

        return np.array([pos_vel, pos_acc, theta_vel, theta_acc])

    def lagr(x, u):
        dx = x_delta(x)
        u = u[np.newaxis]
        return dx.T @ Q @ dx + u.T @ R @ u

    def phi(x):
        dx = x_delta(x)
        return terminal_scale * np.squeeze(dx.T @ Q @ dx)

    def x_delta(x):
        dx = x - x_final
        d_theta = np.mod(dx[2] + np.pi, 2 * np.pi) - np.pi
        # d_theta = np.arccos(np.cos(dx[2]))
        return np.array([[dx[0], dx[1], d_theta, dx[3]]]).T

    u_opt, costs = ddp(x_init, n, m, phi, lagr, F, dt)
    plot_costs(costs)


if __name__ == '__main__':
    main()
