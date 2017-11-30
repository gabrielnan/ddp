import autograd.numpy as np
from cartpolesystem import CartpoleSystem, CartpoleCost
from ddp import ddp, plot_costs


def main():
    # Define System
    g = -9.81  # gravitational acceleration
    m_car = 2  # mass of car
    m_ball = 1  # mass of pole tip
    length = 0.5  # pole length
    system = CartpoleSystem(m_car, m_ball, length, g)

    # DDP var init
    dt = .01  # time step
    time_horizon = 3 # in seconds
    T = int(time_horizon / dt)  # time steps horizon
    m = 1  # control dim
    n = 4  # state dim
    x_init = np.array([.0, .0, np.pi/8, .0])
    # u_bar = np.random.rand(T, m) * 10
    u_bar = np.zeros([T, m])

    # Define cost
    # x_final = np.array([.0, .0, np.pi, .0])
    x_final = np.array([.0, .0, .0, .0])
    Q = np.diag([0, 5, 20, 20])
    R = np.array([[0.1]])
    terminal_scale = T/10
    cost = CartpoleCost(x_final, terminal_scale, Q, R)

    u_opt, costs = ddp(x_init, n, m, cost, system, dt, T, max_iters=20,
                       u_bar=u_bar)
    system.vis(u_opt, dt, x_init, filename='cartpole_opt')
    plot_costs(costs)
    # u = np.zeros(T)
    # system.vis(u, dt, np.array([0.0, 0.0, np.pi / 2, 0.0]))


if __name__ == '__main__':
    main()
