import math
import warnings
from itertools import starmap, repeat, zip_longest
from functools import partial
from collections import deque
from operator import attrgetter

import numpy as np

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
    x_u_proj = B.T.dot(P_t1).dot(A)
    K_t = np.linalg.solve(R + B.T.dot(P_t1).dot(B), x_u_proj)
    P_t = Q + A.T.dot(P_t1).dot(A) - x_u_proj.T.dot(K_t)
    return P_t, K_t

def affine_backpropagation(Q, s, R, z, A, B, P, o):
    """
    minimizeᵤ ∑ₜ uₜRₜuₜ + 2 zₜᵀ uₜ + xₜQₜxₜ + 2 sₜᵀ xₜ
    s.t.          xₜ₊₁ = A xₜ₊₁ + B uₜ

    returns Pₜ, oₜᵀ, Kₜ, kₜ

    s.t.
    optimal uₜ = - Kₜxₜ - kₜ

    and

    Value function is quadratic
    xₜPₜxₜ + 2 oₜᵀ xₜ ≡ ∑ₖ₌ₜᵀ uₖRuₖ + 2 zₖᵀ uₖ + xₖQxₖ + 2 sₖᵀ xₖ
                        s.t. xₖ₊₁ = A xₖ₊₁ + B uₖ
    """
    # Solution:
    # (1) Kₜ = (R + BᵀPₜ₊₁B)⁻¹BᵀPₜ₊₁Aₜ
    # (2) Pₜ = Qₜ + AₜᵀPₜ₊₁Aₜ -     AₜᵀPₜ₊₁BKₜ
    # (3) oₜ = sₜ + Aₜᵀoₜ₊₁    - Kₜᵀ(zₜ + Bᵀoₜ₊₁)
    # (4) kₜ = (R + BᵀPₜ₊₁B)⁻¹(zₜ + Bᵀoₜ₊₁)

    # Eq(1)
    G = R + B.T.dot(P).dot(B)
    K = np.linalg.solve(G, B.T.dot(P).dot(A))
    # Eq(2)
    P_new = Q + A.T.dot(P).dot(A) - A.T.dot(P).dot(B).dot(K)
    # Eq(3)
    o_new = s + A.T.dot(o) - K.T.dot(z + B.T.dot(o))
    # Eq(4)
    k = np.linalg.solve(G, z + B.T.dot(o))
    return P_new, o_new, K, k


def lqr_value_control(P_t1, A, B, Q, R):
    P_t, K_t = discrete_algebric_ricatti_eq(P_t1, A, B, Q, R)
    return P_t, K_t

def repeat_maybe_inf(a, T):
    return (repeat(a)
            if math.isinf(T)
            else repeat(a, int(T)))


def _check_shape(name, X, shape_expected):
    shapes = (map(attrgetter('shape'), X)
                if isinstance(X, list)
                else [X.shape])
    if any(s != shape_expected for s in shapes):
        raise ValueError("Bad shape={} for {}. Expected {}".format(
            s, name, shape_expected))

class LinearSystem:
    """
    minimizeᵤ ∑ₜ uₜRₜuₜ + 2 zₜᵀ uₜ + xₜQₜxₜ + 2 sₜᵀ xₜ
    s.t.          xₜ₊₁ = A xₜ₊₁ + B uₜ
    """
    def __init__(self, A, B, Q, s, R, z, Q_T, s_T, T):
        xD = A.shape[1]
        uD = B.shape[1]
        if not A.shape == (xD, xD): raise ValueError()
        if not B.shape == (xD, uD): raise ValueError()
        if not Q_T.shape == (xD, xD): raise ValueError()
        if not s_T.shape == (xD,): raise ValueError()

        _check_shape("Q", Q, (xD, xD))
        Qs_rev = Q if isinstance(Q, list) else repeat_maybe_inf(Q, T)

        _check_shape("s", s, (xD,))
        ss_rev = s if isinstance(s, list) else repeat_maybe_inf(s, T)

        _check_shape("R", R, (uD, uD))
        Rs_rev = R if isinstance(R, list) else repeat_maybe_inf(R, T)

        _check_shape("z", z, (uD, ))
        zs_rev = z if isinstance(z, list) else repeat_maybe_inf(z, T)

        self.A = A
        self.B = B
        self.Qs_rev = Qs_rev
        self.Q_T = Q_T
        self.ss_rev = ss_rev
        self.s_T = s_T
        self.Rs_rev = Rs_rev
        self.zs_rev = zs_rev
        self.T = T

    def f(self, x_t, u_t, t):
        A = self.A
        B = self.B
        return (x_t
                if t >= self.T
                else A.dot(x_t) + B.dot(u_t))

    def costs(self, xs, us):
        us_rev = reversed(us)
        ctrl_costs = [u_t.T.dot(R).dot(u_t) + 2 * z.T.dot(u_t)
                      for u_t, R, z in zip(us_rev, self.Rs_rev, self.zs_rev)]
        xs_rev = reversed(xs)
        state_costs = [x_t.T.dot(Q).dot(x_t) + 2 * s.dot(x_t)
                       for x_t, Q, s in zip(xs_rev, self.Qs_rev, self.ss_rev)]
        return (c + s for c, s in zip_longest( reversed(ctrl_costs),
                                               reversed(state_costs),
                                               fillvalue=0))

    def lqr_forward_pt(self, x0, us=None):
        xs = [x0]
        if us is None:
            us = np.random.rand(self.T)
        for t in range(self.T):
            xs.append(self.f(xs[t], us[t]))

        return xs, us

    def lqr_backward_pt(self, xs, us):
        Ps = [self.Q_T]
        for t in range(self.T-1, 0, -1):
            us[t] = lqr_control(Ps[0], self.A, self.B, self.Rs_rev[t], xs[t])
            Ps.insert(
                0,
                lqr_value(Ps[0], self.A, self.B, self.Qs_rev[t], self.Rs_rev[t],
                          xs[t], us[t])
            )
        return us

    def solve(self, x0, traj_len=100, max_iter=1000, ε=1e-4):
        if not x0.shape[0] == self.A.shape[0]: raise ValueError()
        P_t = self.Q_T
        traj_len = int(min(self.T, traj_len))
        Ps = deque([P_t], traj_len)
        os = deque([self.s_T], traj_len)
        Ks = deque([], traj_len)
        ks = deque([], traj_len)
        T = min(self.T, max_iter)
        # backward
        for t, Q, s, R, z in zip(reversed(range(T)),
                                  self.Qs_rev, self.ss_rev,
                                  self.Rs_rev, self.zs_rev):
            P_t, o_t, K_t, k_t = affine_backpropagation(
                Q, s, R, z, self.A, self.B, Ps[-1], os[-1])
            if (np.linalg.norm(P_t[:] - Ps[-1][:])
                + np.linalg.norm(o_t - os[-1]) < ε):
                break
            Ps.appendleft(P_t)
            os.appendleft(o_t)
            Ks.appendleft(K_t)
            ks.appendleft(k_t)

        xs = [x0]
        us = [-Ks[0].dot(xs[0]) - ks[0]]
        # forward
        if math.isinf(self.T):
            Ks = repeat(Ks[0])
            ks = repeat(ks[0])

        for t, K, k in zip(range(traj_len), Ks, ks):
            xs.append(self.f(xs[t], us[t], t))
            if t+1 < self.T:
                us.append(-K.dot(xs[t+1]) - k)
        return xs, us


def quadrotor_linear_system(m=1,
                            r0=10,
                            A = [[1, 1],
                                 [0, 1]],
                            Q = [[1, 0],
                                 [0, 0]],
                            T = float('inf')):
    # http://www.argmin.net/2018/02/08/lqr/
    # u_t = u_t - g
    # x_t = [position; velocity]
    B = [[0],
         [1/m]]
    R = [[r0]]
    Q_T = Q
    def plotables(xs, us, linsys, traj_len):
        costs = list(linsys.costs(xs, us))[:traj_len]
        return [("pos", np.array([x[0] for x in xs[:traj_len]])),
                ("vel", np.array([x[1] for x in xs[:traj_len]])),
                ("ctrl", np.array(us[:traj_len])),
                ("cost", costs)]
    x0 = np.array([-1, 0])
    return [plotables, x0] + list(map(np.array, [A, B, Q, np.zeros_like(x0), R, np.zeros((1,)),
                                                 Q_T, np.zeros_like(x0), T]))


def plot_solution(Ts, ylabel_ydata, axes=None,
                  plot_fn=partial(Axes.plot, label='-')):
    if axes is None:
        print("Creating a new figure")
        fig = plt.figure()
        axes = fig.subplots(2,2).ravel().tolist()
        fig.subplots_adjust(wspace=0.32)

    for ax, (ylabel, ydata) in zip(axes, ylabel_ydata):
        plot_fn(ax, Ts, ydata)
        ax.set_ylabel(ylabel)
        ax.legend()
    return axes[0].figure


def test_quadrotor_linear_system_plot():
    # plot cost, trajectory, control
    fig = None
    traj_len = 30
    for r0 in [1, 10, 100]:
        plotables, x0, *linsys = quadrotor_linear_system(r0=r0)
        quad = LinearSystem(*linsys)
        xs, us = quad.solve(x0, traj_len)
        ylabels_ydata = plotables(xs, us, quad, traj_len)
        fig = plot_solution(np.arange(traj_len), ylabels_ydata,
                            axes=None if fig is None else fig.axes,
                            plot_fn=partial(Axes.plot,
                                            label='{}'.format(r0)))
    plt.show()


if __name__ == '__main__':
    test_quadrotor_linear_system_plot()
