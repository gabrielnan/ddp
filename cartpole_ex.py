import autograd.numpy as np
from cartpole import CartpoleSystem, CartpoleCost
from ddp import ddp
from util import plot_costs, run_dynamics


def main():
    # Define System
    g = -9.81  # gravitational acceleration
    m_car = 1  # mass of car
    m_pole = .1  # mass of pole tip
    length = 0.4  # pole length
    system = CartpoleSystem(m_car, m_pole, length, g)

    # DDP var init
    dt = .01  # time step
    time_horizon = 2.1  # in seconds
    # time_horizon = 3  # in seconds
    T = int(time_horizon / dt)  # time steps horizon
    num_iters = 100  # number of iterations
    m = 1  # control dim
    n = 4  # state dim
    x_init = np.array([.0, .0, .0, .0])
    # u_bar = np.random.rand(T, m) * 10
    u_bar = np.zeros([T, m])

    # Define cost
    x_final = np.array([.0, .0, np.pi, .0])
    # x_final = np.array([.0, .0, .0, .0])
    Q = np.diag([10, 0, 40, .1])
    R = np.array([[.04]])
    terminal_scale = 10
    cost = CartpoleCost(x_final, terminal_scale, Q, R)

    num = 2

    x_opt, u_opt, costs = ddp(x_init, n, m, cost, system, dt, T,
                              max_iters=num_iters, u_bar=u_bar)
    system.vis(x_opt, dt, filename='cartpole_opt' + str(num))
    system.plot_states(x_opt, x_final, filename='states' + str(num))
    plot_costs(costs, filename='cost' + str(num))
    # u = np.zeros(2* T)
    # x, _ = run_dynamics(u, system.F, dt, np.array([0.0, 0.0, np.pi-0.1, 0.0]))
    # system.vis(x, dt)


if __name__ == '__main__':
    main()
