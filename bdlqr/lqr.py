import math
import warnings
from itertools import starmap, repeat, zip_longest
from functools import partial
from collections import deque
from operator import attrgetter
from logging import getLogger, DEBUG, INFO, basicConfig
basicConfig()
LOG = getLogger(__name__)
LOG.setLevel(DEBUG)

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from bdlqr.linalg import ScalarQuadFunc, AffineFunction


def affine_backpropagation(Q, s, R, z, A, B, P, o, γ=1):
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
    P = γ*P
    o = γ*o
    G = R + B.T.dot(P).dot(B)
    K = np.linalg.solve(G, B.T.dot(P).dot(A))
    # Eq(2)
    P_new = Q + A.T.dot(P).dot(A) - A.T.dot(P).dot(B).dot(K)
    # Eq(3)
    o_new = s + A.T.dot(o) - K.T.dot(z + B.T.dot(o))
    # Eq(4)
    k = np.linalg.solve(G, z + B.T.dot(o))
    return P_new, o_new, K, k


def affine_backpropagation2(Q, s, R, z, A, B, P, o, γ=1):
    """
    minimizeᵤ uₜRₜuₜ + 2 zₜᵀ uₜ + xₜQₜxₜ + 2 sₜᵀ xₜ + xₜ₊₁ᵀ Pₜ₊₁ xₜ₊₁ + 2 oₜᵀxₜ₊₁
    s.t.          xₜ₊₁ = A xₜ + B u

    returns Pₜ, oₜᵀ, Kₜ, kₜ

    >>> xD = np.random.randint(100)
    >>> uD = np.random.randint(100)
    >>> Qsqrt = np.random.rand(xD, xD)
    >>> Q = Qsqrt.T.dot(Qsqrt)
    >>> s = np.random.rand(xD)
    >>> Rsqrt = np.random.rand(uD, uD)
    >>> R = Rsqrt.T.dot(Rsqrt)
    >>> z = np.random.rand(uD)
    >>> A = np.random.rand(xD, xD)
    >>> B = np.random.rand(xD, uD)
    >>> Psqrt = np.random.rand(xD, xD)
    >>> P = Psqrt.T.dot(Psqrt)
    >>> o = np.random.rand(xD)
    >>> P_, o_, K, k = affine_backpropagation2(Q, s, R, z, A, B, P, o)
    >>> P2, o2, K2, k2 = affine_backpropagation(Q, s, R, z, A, B, P, o)
    >>> np.allclose(P_, P2)
    True
    >>> np.allclose(o_, o2)
    True
    >>> np.allclose(K, K2)
    True
    >>> np.allclose(k, k2)
    True
    """
    P = γ*P
    o = γ*o
    dynamics = AffineFunction(np.hstack((A, B)), np.zeros(A.shape[0]))
    stage_cost = ScalarQuadFunc(Q, s, 0).add_concat(ScalarQuadFunc(R, z, 0))
    terminal_cost = dynamics.dot(P * dynamics) + 2. * o * dynamics
    cost = stage_cost + terminal_cost
    xD = s.shape[0]
    uD = z.shape[0]
    cost_u = cost.partial_f(xD, xD+uD)
    uopt = cost_u.argmin()
    K = - uopt.A
    k = - uopt.B
    cost_opt = cost_u(uopt)
    P_new = cost_opt.Q
    o_new = cost_opt.l
    return P_new, o_new, K, k


def repeat_maybe_inf(a, T):
    return (repeat(a)
            if math.isinf(T) or np.isinf(T)
            else [a] * int(T))


def _check_shape(name, X, shape_expected, listlen):
    if isinstance(X, list) and len(X) != listlen:
        raise ValueError("Bad length for {}. Expected {}".format(
            name, listlen))
    shapes = (map(attrgetter('shape'), X)
                if isinstance(X, list)
                else [X.shape])
    if any(s != shape_expected for s in shapes):
        raise ValueError("Bad shape for {}. Expected {}".format(
            name, shape_expected))

class LinearSystem:
    """
    minimizeᵤ ∑ₜ uₜRₜuₜ + 2 zₜᵀ uₜ + xₜ₊₁Qₜ₊₁xₜ₊₁ + 2 sₜ₊₁ᵀ xₜ₊₁
    s.t.          xₜ₊₁ = A xₜ₊₁ + B uₜ
    """
    def __init__(self, A, B, Q, s, R, z, Q_T, s_T, T, γ=1.):
        xD = A.shape[1]
        uD = B.shape[1]
        if not A.shape == (xD, xD): raise ValueError()
        if not B.shape == (xD, uD): raise ValueError()
        if not Q_T.shape == (xD, xD): raise ValueError()
        if not s_T.shape == (xD,): raise ValueError()

        _check_shape("Q", Q, (xD, xD), T-1)
        Qs_rev = (list(reversed(Q)) + [np.zeros_like(Q_T)]
                  if isinstance(Q, list)
                  else repeat_maybe_inf(Q, T))

        _check_shape("s", s, (xD,), T-1)
        ss_rev = (list(reversed(s)) + [np.zeros_like(s_T)]
                  if isinstance(s, list)
                  else repeat_maybe_inf(s, T))

        _check_shape("R", R, (uD, uD), T)
        Rs_rev = (list(reversed(R))
                  if isinstance(R, list)
                  else repeat_maybe_inf(R, T))

        _check_shape("z", z, (uD, ), T)
        zs_rev = (list(reversed(z))
                  if isinstance(z, list)
                  else repeat_maybe_inf(z, T))

        self.A = A
        self.B = B
        self.Qs_rev = Qs_rev
        self.Q_T = Q_T
        self.ss_rev = ss_rev
        self.s_T = s_T
        self.Rs_rev = Rs_rev
        self.zs_rev = zs_rev
        self.γ = γ
        self.T = T

    def f(self, x_t, u_t, t):
        A = self.A
        B = self.B
        return (x_t
                if t >= self.T
                else A.dot(x_t) + B.dot(u_t))

    def costs(self, xs, us):
        ctrl_costs = [0] + [u_t.T.dot(R).dot(u_t) + 2 * z.T.dot(u_t)
                      for u_t, R, z in zip(reversed(us), self.Rs_rev, self.zs_rev)]
        assert len(ctrl_costs) == len(us) + 1
        state_costs = [x_t.T.dot(Q).dot(x_t) + 2 * s.dot(x_t)
                       for x_t, Q, s in zip(reversed(xs), self.Qs_rev, self.ss_rev)] + [0]
        assert len(state_costs) == len(xs) + 1
        return (c + s for c, s in zip( reversed(ctrl_costs),
                                       reversed(state_costs) ))

    def solve_f(self, t=0, traj_len=100, max_iter=1000, ε=1e-6, return_min=False):
        P_t = self.Q_T
        eff_backprop = int(min(self.T - t, max_iter))
        Ps = deque([P_t], eff_backprop)
        os = deque([self.s_T], eff_backprop)
        Ks = deque([], eff_backprop)
        ks = deque([], eff_backprop)
        # backward
        for t, Q, s, R, z in zip(reversed(range(eff_backprop)),
                                 self.Qs_rev, self.ss_rev,
                                 self.Rs_rev, self.zs_rev):
            P_t, o_t, K_t, k_t = affine_backpropagation(
                Q, s, R, z, self.A, self.B, Ps[0], os[0], γ=self.γ)
            Ps.appendleft(P_t)
            os.appendleft(o_t)
            Ks.appendleft(K_t)
            ks.appendleft(k_t)

        # forward
        if math.isinf(self.T):
            Ks = repeat(Ks[0])
            ks = repeat(ks[0])
        us = (AffineFunction(-Kt, -kt) for Kt, kt in zip(Ks, ks))
        Vs = (ScalarQuadFunc(Pt, ot, np.array(0))
              for Pt, ot in zip(Ps, os))
        return ((us, Vs)
                if return_min
                else us)

    def solve(self, x0, traj_len=100, max_iter=1000, ε=1e-6, return_min=False):
        if not x0.shape[0] == self.A.shape[0]: raise ValueError()
        ufs, Vfs = self.solve_f(traj_len=traj_len, max_iter=max_iter, ϵ=ϵ, return_min=True)
        xs = [x0]
        us = []
        eff_traj_len = min(self.T, traj_len)
        for t, uft in zip(range(eff_traj_len), ufs):
            us.append(uft(xs[t]))
            xs.append(self.f(xs[t], us[t], t))

        assert len(us) == eff_traj_len
        assert len(xs[1:]) == eff_traj_len
        return ((xs[1:], us, next(Vfs)(x0))
                if return_min
                else (xs[1:], us))

    def __repr__(self):
        return "LinearSystem({s.A}, {s.B}, {s.Q}, {s.s}, {s.R}, {s.z}, {s.Q_T}, {s.s_T}, {s.T}, {s.γ})".format(s=self)



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
        LOG.info("Creating a new figure")
        fig = plt.figure()
        axes = fig.subplots(2,2).ravel().tolist()
        fig.subplots_adjust(wspace=0.32)

    legend_plotted = False
    for ax, (ylabel, ydata) in zip(axes, ylabel_ydata):
        plot_fn(ax, Ts, ydata)
        ax.set_ylabel(ylabel)
        if not legend_plotted:
            ax.legend()
            legend_plotted = True
    return axes[0].figure


def test_quadrotor_linear_system_plot(T=float('inf')):
    # plot cost, trajectory, control
    fig = None
    traj_len = 30
    for r0 in [1, 10, 100]:
        plotables, x0, *linsys = quadrotor_linear_system(r0=r0, T=T)
        quad = LinearSystem(*linsys)
        xs, us = quad.solve(x0, traj_len)
        ylabels_ydata = plotables(xs, us, quad, traj_len)
        fig = plot_solution(np.arange(traj_len), ylabels_ydata,
                            axes=None if fig is None else fig.axes,
                            plot_fn=partial(Axes.plot,
                                            label='{}'.format(r0)))
    plt.show()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    test_quadrotor_linear_system_plot()
    test_quadrotor_linear_system_plot(T=40)
