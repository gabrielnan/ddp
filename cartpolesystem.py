import os.path

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd.numpy import sin, cos
from matplotlib import animation

from cost import Cost
from ddp import run_dynamics
from system import System


class CartpoleSystem(System):
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

    def vis(self, u, dt, x_init=None, filename='cartpole', path='vis',
            file_ext='mp4', show=False):
        """Visualization of cartpole system given control

        :param dt: time step
        :param x_init: initial state
        :param u: control sequence
        :param file_ext: video file extension
        :param path: video file path
        :param filename: video filename
        :param show: boolean to show animation
        """
        if x_init is None:
            if self.x_init is None:
                raise ValueError('Must provide an initial state')
            x_init = self.x_init

        x, _ = run_dynamics(u, self.F, dt, x_init)
        cart_pos = np.array([x[:, 0], np.zeros(len(x))]).T
        ball_pos = self.ball_pos(x)

        # setup figure and animation
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-1, 1),
                             ylim=(-self.length*2, self.length*2))
        ax.grid()
        line, = ax.plot([], [], 'o-', lw=2)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def init():
            """initialize animation"""
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(i):
            pos = np.array([cart_pos[i], ball_pos[i]]).T
            line.set_data(*pos)
            # time_text.set_text('time = %.3f' % dt * i)
            return line, time_text

        from time import time
        t0 = time()
        animate(0)
        t1 = time()
        interval = 1000 * dt - (t1 - t0)

        ani = animation.FuncAnimation(fig, animate, frames=len(u),
                                      interval=interval, blit=True,
                                      init_func=init)
        if not os.path.exists(path):
            os.makedirs(path)
        full_filename = os.path.join(path, filename + '.' + file_ext)
        ani.save(full_filename, extra_args=['-vcodec', 'libx264'])
        if show:
            plt.show()

    def ball_pos(self, x):
        x = x.T
        x_pos = x[0] + sin(x[2]) * self.length
        y_pos = -cos(x[2]) * self.length
        return np.array([x_pos, y_pos]).T


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


