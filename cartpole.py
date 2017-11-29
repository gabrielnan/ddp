import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd.numpy import sin, cos

from cost import Cost
from ddp import ddp, plot_costs, run_dynamics
from system import System


class Cartpole(System):
    """Cartpole system with constants
    """

    def __init__(self, m_cart, m_ball, length, g=-9.8, x_init=None):
        """
        :param m_cart: mass of cart (kg)
        :param m_ball: mass of ball (kg)
        :param length: length of pole (m)
        :param g: gravitational acceleration (m / s^2)
        """
        self.x_init = x_init
        self.m_car = m_cart
        self.m_ball = m_ball
        self.length = length
        self.g = g
        self.x_dim = 4
        self.x_description = ['pos', 'pos_vel', 'theta', 'theta_vel']

    def F(self, x, u):
        # Rename system constants for readibility
        m1 = self.m_car
        m2 = self.m_ball
        l = self.length
        g = self.g

        # Extract state values for readibility
        pos_vel = x[1]
        theta = x[2]
        theta_vel = x[3]

        pos_acc = (-m2 * g * cos(theta) * sin(theta) + u \
                   + m2 * theta_vel ** 2 * sin(theta)) \
                  / (m1 + m2 - m2 * cos(theta) ** 2)
        theta_acc = (g * sin(theta) - cos(theta) * pos_acc) / l

        return np.array([pos_vel, pos_acc, theta_vel, theta_acc])

    # def vis(self, u, dt, x_init=None):
    #     """Visualization of cartpole system given control
    #
    #     :param dt: time step
    #     :param x_init: initial state
    #     :param u: control sequence
    #     """
    #     if x_init is None:
    #         if self.x_init is None:
    #             raise ValueError('Must specify an initial state')
    #         x_init = self.x_init
    #
    #     x = run_dynamics(u, self.F, dt, x_init)
    #     pos = self.ball_pos(x)
    #
    #     # setup figure and animation
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
    #                          xlim=())

    def ball_pos(self, x):
        x = x.T
        x_pos = x[0] + sin(x[2]) * self.length
        y_pos = -cos(x[2]) * self.length
        return x_pos, y_pos


class CartpoleCost(Cost):
    def __init__(self, x_final, terminal_scale, Q, R):
        self.x_final = x_final
        self.terminal_scale = terminal_scale
        self.Q = Q
        self.R = R

    def lagr(self, x, u):
        Q = self.Q
        R = self.R
        dx = self.x_delta(self.x_final, x)
        u = u[np.newaxis]
        return np.squeeze(dx.T @ Q @ dx + u.T @ R @ u)

    def phi(self, x):
        Q = self.Q
        dx = self.x_delta(self.x_final, x)
        return self.terminal_scale * np.squeeze(dx.T @ Q @ dx)

    @staticmethod
    def x_delta(x1, x2):
        dx = x1 - x2
        d_theta = np.mod(dx[2] + np.pi, 2 * np.pi) - np.pi
        # d_theta = np.arccos(cos(dx[2])
        return np.array([[dx[0], dx[1], d_theta, dx[3]]]).T


def main():
    # Define System
    g = -9.8  # gravitational acceleration
    m_car = 0.5  # mass of car
    m_ball = 0.2  # mass of pole tip
    length = 3  # pole length
    system = Cartpole(m_car, m_ball, length, g)

    # Define cost
    x_final = np.array([.0, .0, np.pi, .0])
    Q = np.diag([1, 5, 5, 3])
    R = np.array([[1]])
    terminal_scale = 10
    cost = CartpoleCost(x_final, terminal_scale, Q, R)

    # DDP var init
    x_init = np.array([.0, .0, .0, .0])
    dt = .005  # time step
    T = int(3 / dt)  # time horizon
    m = 1  # control dim
    n = 4  # state dim

    u_opt, costs = ddp(x_init, n, m, cost, system, dt, T)
    plot_costs(costs)


if __name__ == '__main__':
    main()
