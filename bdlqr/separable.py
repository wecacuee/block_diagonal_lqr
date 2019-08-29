import warnings
import math
from os.path import join as pjoin
from functools import partial
from collections import namedtuple, deque
from operator import attrgetter
from itertools import zip_longest, product
from logging import getLogger, INFO, DEBUG, basicConfig
basicConfig()
LOG = getLogger(__name__)
LOG.setLevel(INFO)

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

from matplotlib.axes import Axes
from kwplus.functools import recpartial

from bdlqr.lqr import (LinearSystem, quadrotor_linear_system, plot_solution,
                       affine_backpropagation)
from bdlqr.diff_substr import diff_substr
from bdlqr.admm import admm
from bdlqr.linalg import randps, ScalarQuadFunc, AffineFunction
from bdlqr.functoolsplus import getname, list_extendable


def solve_seq(slsys, y0, x0, traj_len):
    """
    v*_{1:T} = minimize_v ∑ₜ yₜQₜyₜ
    s.t.          yₜ₊₁ = Ay yₜ + Bv vₜ

    minimize_u ∑ₜ uₜRuₜ + |E xₜ - vₜ*|₂²
    s.t.          xₜ₊₁ = Ax xₜ₊₁ + Bu uₜ
    """
    Qy  = slsys.Qy
    R   = slsys.R
    Ay  = slsys.Ay
    Bv  = slsys.Bv
    QyT = slsys.QyT
    E   = slsys.E
    Ax  = slsys.Ax
    Bu  = slsys.Bu
    T   = slsys.T

    vD = Bv.shape[-1]
    y_sys = LinearSystem(Ay, Bv, Qy, np.zeros(Qy.shape[0]), np.zeros((vD, vD)),
                         np.zeros(vD), QyT, np.zeros(QyT.shape[0]), T)
    v0 = E.dot(x0)
    y1  = y_sys.f(y0, v0, 0)
    ys, vs = y_sys.solve(y1, traj_len)
    assert len(vs) == traj_len
    assert len(ys) == traj_len

    # Reformulate the affine to a linear system by making the state a
    # homogeneous vector
    # minimize_u ∑ₜ uₜRuₜ + xhₜ Qxh xhₜ
    # s.t.          xhₜ₊₁ = Axh xhₜ₊₁ + Buh uₜ
    # where Qxh = [  Eᵀ ] [ E  -vₜ]
    #             [ -vᵀ ]
    #  and  Axh = [ A   0 ]
    #             [ 0   1 ]
    #  and  Buh = [ Bu ]
    #             [ 0  ]
    #  and xhₜ =  [ xₜ ]
    #             [ 1  ]
    x0h = np.hstack((x0, 1))
    E_vs = [np.hstack((E, -v.reshape(-1, 1)))
            for v in vs]
    Qsx = [E_vt.T.dot(E_vt)
           for E_vt in E_vs]
    Axh = np.eye(x0h.shape[0])
    Axh[:-1, :-1] = Ax
    uD = Bu.shape[-1]
    Buh = np.vstack((Bu, np.zeros((1, uD))))
    x_sys = LinearSystem(Axh, Buh,
                         Qsx[:-1], np.zeros(Qsx[0].shape[0]),
                         R, np.zeros(R.shape[0]),
                         Qsx[-1], np.zeros(Qsx[-1].shape[0]), len(vs))
    xhs, us = x_sys.solve(x0h, traj_len)
    assert len(us) == traj_len
    xs = [xh[:-1] for xh in xhs]
    assert len(xs) == traj_len

    # Generate forward trajectory
    vs = [E.dot(xt) for xt in xs]
    ys = [y0]
    for t, vt in enumerate(vs):
        yt = y_sys.f(ys[-1], vt, t)
        ys.append(yt)
    assert len(xs) == len(us)
    assert len(ys[1:]) == len(xs)
    return ys[1:], xs, us


def joint_linear_system(slsys, y0, x0):
    slsys = slsys
    Qy  = slsys.Qy
    R    = slsys.R
    Ay  = slsys.Ay
    Bv   = slsys.Bv
    QyT = slsys.QyT
    E    = slsys.E
    Ax   = slsys.Ax
    Bu   = slsys.Bu
    T    = slsys.T
    # Q = [Qy  0 ]
    #     [ 0   0 ]
    # Xₜ = [yₜ]
    #      [xₜ]
    # A = [Ay Bv E ]
    #     [ 0  Ax  ]
    #
    # B = [ 0  ]
    #     [ Bu ]
    X0 = np.hstack((y0, x0))
    Q = np.vstack((np.hstack((Qy, np.zeros((Qy.shape[0], x0.shape[0])))),
                   np.zeros((x0.shape[0], X0.shape[0]))))
    QT = np.vstack((np.hstack((QyT, np.zeros((QyT.shape[0], x0.shape[0])))),
                    np.zeros((x0.shape[0], X0.shape[0]))))
    A = np.vstack((np.hstack((Ay, Bv.dot(E))),
                    np.hstack((np.zeros((Ax.shape[0], Ay.shape[1])), Ax))))
    B = np.vstack((np.zeros((y0.shape[0], Bu.shape[1])),
                   Bu))
    return A, B, Q, R, QT, T, X0


def solve_full(slsys, y0, x0, traj_len):
    A, B, Q, R, QT, T, X0 = joint_linear_system(slsys, y0, x0)
    Xs, us = LinearSystem(A, B,
                          Q, np.zeros(Q.shape[0]),
                          R, np.zeros(R.shape[0]),
                          QT, np.zeros(QT.shape[0]), T).solve(X0, traj_len)
    # x0 is input to the algorithm
    xs = [X[y0.shape[0]:] for X in Xs]

    # y0 and y1 are predetermined so ignore the first time step
    # but compute the last y
    ys = [X[:y0.shape[0]] for X in Xs[1:]]
    Ay = slsys.Ay
    Bv = slsys.Bv
    E  = slsys.E
    ylast = ys[-1] if len(ys) else y0
    ys.append(Ay.dot(ys[-1]) + Bv.dot(E).dot(xs[-1]))

    assert len(ys) == traj_len
    assert len(xs) == traj_len
    assert len(us) == traj_len
    return ys, xs, us


def solve_admm(slsys, y0, x0, traj_len, ρ=1, admm_=admm):
    """
    Solve the two minimizations alternatively:

    minimize_v ∑ₜ yₜQₜyₜ + wₖₜᵀ(E xₖₜ - vₜ) + 0.5 ρ|E xₖₜ - vₜ|²
    s.t.          yₜ₊₁ = Ay yₜ + Bv vₜ

    minimize_u ∑ₜ uₜRuₜ + wₖₜᵀ(E xₜ - vₖₜ) + 0.5 ρ|E xₜ - vₖₜ|²
    s.t.          xₜ₊₁ = Ax xₜ₊₁ + Bu uₜ

    """
    E = slsys.E
    T = slsys.T
    assert not math.isinf(T)
    _, xs0, us0 = solve_seq(slsys, y0, x0, T)
    us0 = np.vstack(us0)
    vs0 = np.vstack([E.dot(xt) for xt in xs0])
    ws0 = np.vstack([np.zeros_like(vt) for vt in vs0])

    cost_y = slsys.cost_v(y0, vs0)
    cost_u = slsys.cost_u(us0)
    vs, us, ws = admm_((partial(slsys.prox_f_v, y0, x0),
                        partial(slsys.prox_g_u, y0, x0)),
                       vs0, us0, ws0,
                       partial(slsys.constraint_fn, y0, x0), ρ,
                       objs=(partial(slsys.cost_v, y0), slsys.cost_u))
    ys, xs = slsys.forward(y0, x0, us)
    return ys, xs, us


_SeparableLinearSystem = namedtuple('_SeparableLinearSystem',
                                    "Qy R Ay Bv QyT E Ax Bu T γ".split(" "))


class SeparableLinearSystem(_SeparableLinearSystem):
    """
    Represents a linear system problem of the form:

    minimize_u  ∑ₜ yₜQₜyₜ + uₜRuₜ + y_T Q_T y_T
    s.t.          yₜ₊₁ = Ay yₜ + Bv vₜ
                    vₜ = E xₜ
                  xₜ₊₁ = Ax xₜ₊₁ + Bu uₜ
    """
    @classmethod
    def random(cls, yD=1, vD=1, xD=1, uD=1, T=3, rng=np.random):
        return cls(Qy  = randps(yD, rng=rng),
                   R   = randps(uD, rng=rng),
                   Ay  = rng.rand(yD, yD),
                   Bv  = rng.rand(yD, vD),
                   QyT = randps(yD, rng=rng),
                   E   = rng.rand(vD, xD),
                   Ax  = rng.rand(xD, xD),
                   Bu  = rng.rand(xD, uD),
                   T   = T,
                   γ   = 1)

    @classmethod
    def copy(cls, other, **overrides):
        assert isinstance(other, cls)
        odict = other._asdict()
        return cls(**dict(odict, **overrides))

    def f_x(self, x0, u0):
        Ax = self.Ax
        Bu = self.Bu
        return Ax.dot(x0) + Bu.dot(u0)

    def forward_x(self, x0, us):
        xs = [x0]
        for t, ut in enumerate(us):
            xs.append(self.f_x(xs[t], ut))
        return xs[1:]

    def f_y(self, y0, v0):
        Ay = self.Ay
        Bv = self.Bv
        return Ay.dot(y0) + Bv.dot(v0)

    def forward_y(self, y0, vs):
        ys = [y0]
        for t, vt in enumerate(vs):
            ys.append(self.f_y(ys[t], vt))
        return ys[1:]

    def effect(self, x0):
        return self.E.dot(x0)

    def forward(self, y0, x0, us):
        # Generate forward trajectory
        E  = self.E

        xs = self.forward_x(x0, us)
        vs = [E.dot(xt) for xt in xs]
        y1 = self.f_y(y0, E.dot(x0))
        ys = self.forward_y(y1, vs)

        return ys, xs

    def cost_v(self, y0, vs):
        Q = lambda t: self.QyT if t >= self.T else self.Qy
        return sum(yt.T.dot(Q(t)).dot(yt)
                   for t, yt in enumerate(self.forward_y(y0, vs)))

    def cost_u(self, us):
        R = self.R
        return sum(ut.T.dot(R).dot(ut) for ut in us)

    def costs(self, ys, xs, us):
        Q = lambda t: self.QyT if t >= self.T else self.Qy
        R = self.R
        return [yt.T.dot(Q(t)).dot(yt) + (0
                                          if ut is None
                                          else ut.T.dot(R).dot(ut))
                for t, (yt, ut) in enumerate(zip_longest(ys, us))]

    def prox_g_u(self, __, x0, vs, _, ws, ρ):
        """
        return arg min_{us} g(vks - wks / ρ)

        g(vks - wks/ρ) = ∑ uₜᵀRₜuₜ + 0.5 ρ |Exₜ(us) - (vₜ - wₜ/ρ)|₂²
                         s.t. xₜ₊₁ = Ax xₜ + Bu uₜ
        """
        assert len(vs) == len(ws)
        R    = self.R
        E    = self.E
        Ax   = self.Ax
        Bu   = self.Bu
        T    = self.T
        ###
        # ADMM Step 1: Find best us that generate vs
        ###
        # Reformulate the affine to a linear system by making the state a
        # homogeneous vector
        # minimize_u ∑ₜ uₜRuₜ + wₖₜᵀ(E xₜ - vₖₜ) + 0.5 ρ|E xₜ - vₖₜ|²
        # s.t.          xₜ₊₁ = Ax xₜ₊₁ + Bu uₜ
        # where Qxh = [  Eᵀ           ] [ E  -vₖₜ + wₖₜ/ρ]
        #             [ -vₖₜᵀ + wₖₜᵀ/ρ]
        #  and  Axh = [ A   0 ]
        #             [ 0   1 ]
        #  and  Buh = [ Bu ]
        #             [ 0  ]
        #  and xhₜ =  [ xₜ ]
        #             [ 1  ]
        x0h    = np.hstack((x0, 1))
        E_vs    = [np.hstack((E, (-v + w/ρ).reshape(-1,1)))
                   for v, w in zip(vs, ws)]
        Qxs    = [0.5 * ρ * E_vt.T.dot(E_vt)    for E_vt in E_vs]
        Axh     = np.eye(x0h.shape[0])
        Axh[:-1, :-1] = Ax
        Buh     = np.vstack((Bu, np.zeros((1, Bu.shape[1]))))
        x_sys   = LinearSystem(Axh,      Buh,
                               Qxs[:-1], np.zeros(Qxs[0].shape[0]),
                               R,        np.zeros(R.shape[0]),
                               Qxs[-1],  np.zeros(Qxs[-1].shape[0]),
                               len(vs))
        xhs_new, us_new, usmin = x_sys.solve(x0h, len(vs), return_min=True)
        LOG.debug(" xhs[:3]=%s", xhs_new)
        assert len(xhs_new) == len(vs)
        assert len(us_new) == len(vs)
        return np.vstack(us_new)

    def prox_f_v(self, y0, x0, _, us, ws, ρ):
        assert len(us) == len(ws)
        Qy   = self.Qy
        Ay   = self.Ay
        Bv   = self.Bv
        QyT  = self.QyT
        E    = self.E
        T    = self.T
        ###
        # ADMM Step 2: Find best vs that generate are close enough to E x
        ###
        # minimize_v ∑ₜ yₜQₜyₜ + wₖₜᵀ(E xₖₜ - vₜ) + 0.5 ρ|E xₖₜ - vₜ|²
        # s.t.          yₜ₊₁ = Ay yₜ + Bv vₜ
        # Rᵥ = 0.5 ρ I
        # zᵥ = 0.5 ρ ( - E xₖₜ - wₖₜ/ρ )
        xs = self.forward_x(x0, us)
        Rsv = [0.5 * ρ * np.eye(E.shape[0])
               for _ in xs]
        zsv = [0.5 * ρ * ( -E.dot(x) - w/ρ )
               for x, w in zip(xs, ws)]

        # v0 is fixed because x0, v_{1:T} is unknown
        v0 = E.dot(x0)
        y1 = Ay.dot(y0) + Bv.dot(v0)

        y_sys  = LinearSystem(Ay, Bv,
                              Qy, np.zeros(Qy.shape[0]),
                              Rsv, zsv,
                              QyT, np.zeros(QyT.shape[0]), len(us))
        ys_new, vs_new, vsmin = y_sys.solve(y1, len(us), return_min=True)
        LOG.debug(" ys[:3]=%s", ys_new)
        assert len(vs_new) == len(us)
        assert len(ys_new) == len(vs_new)
        return np.vstack(vs_new)

    def constraint_fn(self, y0, x0, vs_us):
        vD = self.Bv.shape[-1]
        vs = vs_us[:, :vD]
        us = vs_us[:, vD:]

        E  = self.E
        assert len(vs) == len(us)
        ulen = len(us)

        xs = self.forward_x(x0, us)
        assert len(xs) == ulen
        # Need to vstack it because it should support the addition operator
        ws = np.vstack(([E.dot(xt) - vt for xt, vt in zip(xs, vs)]))
        assert ws.shape == (len(vs), vs[0].shape[0])
        return ws


def plotables(ys, xs, us, linsys, traj_len):
    costs = linsys.costs(ys, xs, us)[:traj_len]
    return [("pos: y[0]", np.array([y[0] for y in ys[:traj_len]])),
            ("vel: x[0]", np.array([x[0] for x in xs[:traj_len]])),
            ("ctrl: u[0]", np.array(us[:traj_len])),
            ("cost", costs)]


def quadrotor_square_example(γ=1):
    _, y0, Ay, Bv, Qy, _, _, _, QyT, _, _ = quadrotor_linear_system()
    _, x0, Ax, Bu, _, _, R, _, _, _, _ = quadrotor_linear_system()
    E = np.hstack((np.eye(Bv.shape[1]),
                   np.zeros((Bv.shape[1], Ax.shape[1] - Bv.shape[1]))))
    return plotables, y0, x0, Qy, R, Ay, Bv, QyT, E, Ax, Bu, 100, γ


def quadrotor_as_separable(m  = 1,
                           r0 = 1,
                           Ay = [[1.]],
                           Bv = [[1.]],
                           E  = [[1.]],
                           Qy = [[1]],
                           Ax = [[1.]],
                           y0 = [-1],
                           x0 = [0],
                           T  = 30,
                           γ  = 1):
    Bu=[[1/m]]
    R=[[r0]]
    QyT = np.array(Qy)*100
    return [plotables] + list(map(np.asarray, (y0, x0, Qy, R, Ay, Bv, QyT, E, Ax, Bu, T, γ)))


def array2string(arr,
                 arr2str=partial(np.array2string,
                                 separator=",",
                                 suppress_small=True),
                 translate=("\n", " ")):
    return arr2str(arr).translate(str.maketrans(*translate))


def plot_separable_sys_results(example=quadrotor_as_separable, traj_len=30,
                               getsolvers_=list_extendable(
                                   [solve_full, solve_seq, solve_admm]),
                               line_specs='b.- g,-- ro cv- m^- y<- k>-'.split(),
                               plot_dir="/tmp/",
                               arr2str=array2string,
                               fig_string_fmt="Q={Q}, R={R}, x0={x0}, y0={y0}".format):
    plotables, y0, x0, *sepsys = example()
    fig = None
    slsys = SeparableLinearSystem(*sepsys)
    solvers = getsolvers_()
    labels = map(getname, solvers)
    short_labels = diff_substr(labels)
    eff_traj_len = min(slsys.T, traj_len)
    fig_string = fig_string_fmt(Q=arr2str(slsys.Qy),
                                R=arr2str(slsys.R),
                                x0=arr2str(x0),
                                y0=arr2str(y0))
    for solver, label, lnfmt in zip(solvers, short_labels, line_specs):
        ys_full, xs_full, us_full = solver(slsys, y0, x0, eff_traj_len)
        y1 = slsys.Ay.dot(y0) + slsys.Bv.dot(slsys.E).dot(x0)
        ys_full = np.vstack((y0, y1, ys_full))
        xs_full = np.vstack((x0, xs_full))
        fig = plot_solution(
            np.arange(eff_traj_len),
            plotables(ys_full, xs_full, us_full, slsys, eff_traj_len),
            axes= None if fig is None else fig.axes,
            plot_fn=lambda ax, x, y: Axes.plot(ax, x, y, lnfmt, label=label),
            figtitle=fig_string)
    if fig is not None:
        fig.savefig(pjoin(plot_dir, fig_string + ".pdf"))
        fig.show()
        plt.show()


def test_solvers_with_full(seed=None, getsolvers_=list_extendable([solve_admm]),
                           maxD=15, maxT=100):
    """

    >>> test_solvers_with_full()
    True
    """
    if seed is None:
        seed = np.random.randint(100000)
    LOG.info("seed={}".format(seed))
    rng = np.random.RandomState(seed=seed)
    yD, vD, xD, uD = rng.randint(1, maxD+1, size=4)
    T = rng.randint(maxT)
    slsys = SeparableLinearSystem.random(yD=yD, vD=vD, xD=xD, uD=uD, T=T)
    y0 = rng.rand(yD)
    x0 = rng.rand(xD)
    ys_full, xs_full, us_full = solve_full(slsys, y0, x0, T)
    for solver in getsolvers_():
        ys, xs, us = solver(slsys, y0, x0, T)
        assert np.allclose(us, us_full, rtol=1e-2, atol=1e-2)
    return True


if __name__ == '__main__':
    for x0, y0 in product([0.1, 0.0], repeat=2):
        recpartial(
            plot_separable_sys_results,{
                "example.y0": [y0],
                "example.x0": [x0]
            })()
    for r0 in map(np.exp, range(-2, 3)):
        recpartial(
            plot_separable_sys_results,{
                "example.r0": r0
            })()
    test_solvers_with_full()




