import warnings
from itertools import starmap
from functools import partial

import numpy as np
import scipy.linalg

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def lqr_control_pt(P_t1, A, B, R, x_t):
    u_t = np.linalg.solve(R + B.T.dot(P_t1).dot(B), B.T.dot(P_t1).dot(A).dot(x_t))
    return u_t


def lqr_value_pt(P_t1, A, B, Q, R, x_t, u_t=None):
    if u_t is None:
        u_t = lqr_control(P_t1, A, B, R, x_t)
    P_t = Q + A.T.dot(P_t1).A + u_t.T.dot(R + B.T.dot(P_t1).dot(B)).u_t
    return P_t


def discrete_algebric_ricatti_eq(P_t1, A, B, Q, R):
    P_u_inv = np.linalg.inv(R + B.T.dot(P_t1).dot(B))
    x_u_proj = B.T.dot(P_t1).dot(A)
    P_t = Q + A.T.dot(P_t1).dot(A) - x_u_proj.T.dot(P_u_inv).dot(x_u_proj)
    return P_t, P_u_inv, x_u_proj


def lqr_value_control(P_t1, A, B, Q, R):
    P_t, P_u_inv, x_u_proj = discrete_algebric_ricatti_eq(P_t1, A, B, Q, R)
    K_t = P_u_inv.dot(x_u_proj)
    return P_t, K_t


class LinearSystem:
    def __init__(self, A, B, Q, R, Q_T, T):
        xD = A.shape[1]
        uD = B.shape[1]
        if not A.shape == (xD, xD): raise ValueError()
        if not B.shape == (xD, uD): raise ValueError()
        if not Q_T.shape == (xD, xD): raise ValueError()

        Qs = Q if isinstance(Q, list) else [Q]*int(T)
        if any(Q.shape != (xD, xD) for Q in Qs):
            raise ValueError()
        Qs.append(Q_T)

        Rs = R if isinstance(R, list) else [R]*int(T)
        if any(R.shape != (uD, uD) for R in Rs):
            raise ValueError()

        self.A = A
        self.B = B
        self.Qs = Qs
        self.Rs = Rs
        self.T = T

    def f(self, x_t, u_t, t):
        A = self.A
        B = self.B
        return (x_t
                if t >= self.T
                else A.dot(x_t) + B.dot(u_t))

    def cost(self, x_t, u_t, t):
        Q = self.Qs[t]
        return x_t.T.dot(Q).dot(x_t) + (0
                                        if t >= self.T
                                        else u_t.T.dot(self.Rs[t]).dot(u_t))

    def lqr_forward_pt(self, x_0, us=None):
        xs = [x_0]
        if us is None:
            us = np.random.rand(self.T)
        for t in range(self.T):
            xs.append(self.f(xs[t], us[t]))

        return xs, us

    def lqr_backward_pt(self, xs, us):
        Ps = [self.Qs[-1]]
        for t in range(self.T-1, 0, -1):
            us[t] = lqr_control(Ps[0], self.A, self.B, self.Rs[t], xs[t])
            Ps.insert(
                0,
                lqr_value(Ps[0], self.A, self.B, self.Qs[t], self.Rs[t],
                          xs[t], us[t])
            )
        return us

    def solve(self, x_0):
        if not x_0.shape[0] == self.A.shape[0]: raise ValueError()
        P_t = self.Qs[-1]
        Ps = [P_t]
        Ks = []
        #K_t = lqr(self.A, self.B, self.Q, self.R)
        for t in range(self.T-1, 0, -1):
            P_t, K_t = lqr_value_control(P_t, self.A, self.B, self.Qs[t], self.Rs[t])
            Ps.insert(0, P_t)
            Ks.insert(0, K_t)

        us = []
        xs = [x_0]
        for t in range(self.T):
            us.append(-K_t.dot(xs[t]))
            xs.append(self.f(xs[t], us[t], t))
        return xs, us


def quadrotor_linear_system(m=1,
                            r_0=10,
                            A = [[1, 1],
                                 [0, 1]],
                            Q = [[1, 0],
                                 [0, 0]],
                            T = 30):
    # http://www.argmin.net/2018/02/08/lqr/
    # u_t = u_t - g
    # x_t = [position; velocity]
    B = [[0],
         [1/m]]
    R = [[r_0]]
    Q_T = [[100, 0],
           [0, 100]]
    return map(np.array, [A, B, Q, R, Q_T, T])

def plot_solution(cost_fn, xs, us, Ts, axes=None,
                  plot_fn=partial(Axes.plot, label='-')):
    costs = list(starmap(cost_fn, zip(xs[1:], us, Ts)))
    positions = np.array([x[0] for x in xs[1:]])
    velocities = np.array([x[1] for x in xs[1:]])
    if axes is None:
        print("Creating a new figure")
        fig = plt.figure()
        axes = fig.subplots(2,2).ravel().tolist()
        fig.subplots_adjust(wspace=0.32)

    for ax, (ylabel, ydata) in zip(axes,
                                   [("x[0]", positions),
                                    ("x[1]", velocities),
                                    ("u[0]", np.array(us)),
                                    ("cost", costs)]):
        plot_fn(ax, Ts, ydata)
        ax.set_ylabel(ylabel)
        ax.legend()
    return axes[0].figure

def test_quadrotor_linear_system_plot():
    # plot cost, trajectory, control
    quad = LinearSystem(*quadrotor_linear_system())
    xs, us = quad.solve(np.array([-1, 0]))
    fig = plot_solution(quad.cost, xs, us, np.arange(1, quad.T+1))
    plt.show()


if __name__ == '__main__':
    test_quadrotor_linear_system_plot()
