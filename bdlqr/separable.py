import warnings
import math
from functools import partial
from collections import namedtuple, deque
from operator import attrgetter
from itertools import zip_longest
from logging import getLogger, DEBUG, basicConfig
basicConfig()
LOG = getLogger(__name__)
LOG.setLevel(DEBUG)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from bdlqr.full import LinearSystem, quadrotor_linear_system, plot_solution,  affine_backpropagation
from bdlqr.diff_substr import diff_substr
from bdlqr.admm import admm
from bdlqr.linalg import ScalarQuadFunc, AffineFunction


def solve_seq(slsys, y0, x0, traj_len):
    """
    minimize_v ∑ₜ yₜQₜyₜ
    s.t.          yₜ₊₁ = Ay yₜ + Bv vₜ

    minimize_u ∑ₜ uₜRuₜ + |E xₜ - vₜ|₂²
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
    y_sys = LinearSystem(Ay, Bv, Qy, np.zeros(Qy.shape[0]), R,
                         np.zeros(R.shape[0]), QyT, np.zeros(QyT.shape[0]), T)
    v0 = E.dot(x0)
    y1  = y_sys.f(y0, v0, 0)
    ys, vs = y_sys.solve(y1, traj_len)
    assert len(vs) == traj_len
    assert len(ys) == traj_len + 1
    ys.insert(0, y0)
    vs.insert(0, v0)

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
    Buh = np.vstack((Bu, 0))
    x_sys = LinearSystem(Axh, Buh,
                         Qsx[:-1], np.zeros(Qsx[0].shape[0]),
                         R, np.zeros(R.shape[0]),
                         Qsx[-1], np.zeros(Qsx[-1].shape[0]), T)
    xhs, us = x_sys.solve(x0h, traj_len)
    assert len(us) == traj_len
    xs = [xh[:-1] for xh in xhs]
    assert len(xs) == traj_len + 1

    # Generate forward trajectory
    vs = [E.dot(xt) for xt in xs]
    ys = [y0]
    for t, vt in enumerate(vs):
        yt = y_sys.f(ys[-1], vt, t)
        ys.append(yt)
    assert len(xs) == len(us) + 1
    assert len(ys) == len(xs) + 1
    return ys, xs, us


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
    ys = [X[:y0.shape[0]] for X in Xs]
    xs = [X[y0.shape[0]:] for X in Xs]
    return ys, xs, us


def solve_admm(slsys, y0, x0, traj_len, ε=1e-4, ρ=1, max_iter=10):
    """
    Solve the two minimizations alternatively:

    minimize_v ∑ₜ yₜQₜyₜ + wₖₜᵀ(E xₖₜ - vₜ) + 0.5 ρ|E xₖₜ - vₜ|²
    s.t.          yₜ₊₁ = Ay yₜ + Bv vₜ

    minimize_u ∑ₜ uₜRuₜ + wₖₜᵀ(E xₜ - vₖₜ) + 0.5 ρ|E xₜ - vₖₜ|²
    s.t.          xₜ₊₁ = Ax xₜ₊₁ + Bu uₜ

    """
    Qy   = slsys.Qy
    R    = slsys.R
    Ay   = slsys.Ay
    Bv   = slsys.Bv
    QyT  = slsys.QyT
    E    = slsys.E
    Ax   = slsys.Ax
    Bu   = slsys.Bu
    T    = slsys.T
    assert not math.isinf(T)
    ys, xs, us = solve_seq(slsys, y0, x0, T)
    vs = [E.dot(x) for x in xs]
    ws = [np.zeros(vs[0].shape[0]) for _ in range(len(vs))]

    for k in range(max_iter):
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
        E_vs   = [np.hstack((E, (-v + w/ρ).reshape(-1,1)))
                   for v, w in zip(vs, ws)]
        Qxs    = [0.5 * ρ * E_vt.T.dot(E_vt)    for E_vt in E_vs]
        Axh     = np.eye(x0h.shape[0])
        Axh[:-1, :-1] = Ax
        Buh     = np.vstack((Bu, np.zeros((1, Bu.shape[1]))))
        x_sys   = LinearSystem(Axh, Buh,
                               Qxs[:-1], np.zeros(Qxs[0].shape[0]),
                               R, np.zeros(R.shape[0]),
                               Qxs[-1], np.zeros(Qxs[-1].shape[0]),
                               T)
        xhs_new, us_new, usmin = x_sys.solve(x0h, T, return_min=True)
        xs_new = [xh[:-1] for xh in xhs_new]

        ###
        # ADMM Step 2: Find best vs that generate are close enough to E x
        ###
        # minimize_v ∑ₜ yₜQₜyₜ + wₖₜᵀ(E xₖₜ - vₜ) + 0.5 ρ|E xₖₜ - vₜ|²
        # s.t.          yₜ₊₁ = Ay yₜ + Bv vₜ
        # Rᵥ = 0.5 ρ I
        # zᵥ = 0.5 ρ ( - E xₖₜ + wₖₜ/ρ )
        Rsv = [0.5 * ρ * np.eye(E.shape[0])
               for _ in vs]
        zsv = [0.5 * ρ * ( -E.dot(x) - w/ρ )
               for x, w in zip(xs_new, ws)]

        # v0 is fixed because x0, v_{1:T} is unknown
        v0 = E.dot(x0)
        y1 = Ay.dot(y0) + Bv.dot(v0)

        y_sys  = LinearSystem(Ay, Bv,
                              Qy, np.zeros(Qy.shape[0]),
                              Rsv, zsv,
                              QyT, np.zeros(QyT.shape[0]), T)
        ys_new, vs_new, vsmin = y_sys.solve(y1, T, return_min=True)
        ys_new.insert(0, y0)
        vs_new.insert(0, v0)

        ###
        # ADMM Step 3: Update Lagrange parameters
        ###
        ws_new = [w + ρ * (E.dot(x_new) - v_new)
                  for w, x_new, v_new in zip(ws, xs_new, vs_new)]

        change = sum(np.linalg.norm(u - u_new)
                     for u, u_new in zip(us, us_new))

        ys = ys_new
        vs = vs_new
        xs = xs_new
        us = us_new
        ws = ws_new

        ys_f, xs_f =  slsys.forward(x0, y0, us)
        cost_y = sum(y.dot(Qy).dot(y) for y in ys_f)
        LOG.debug("y cost {:.03f}".format(cost_y))
        cost_u = sum(u.dot(R).dot(u) for u in us)
        LOG.debug("u cost {:.03f}".format(cost_u))
        cost_vs = sum(np.linalg.norm(v - E.dot(x))
                      for v, x in zip(vs, xs_f))
        LOG.debug("const. cost {:.03f}".format(cost_vs))
        if change < ε:
            LOG.debug("change {:.03f} => Breaking".format(change))
            break
        else:
            LOG.debug("change {:.03f} => Continuing".format(change))

    return ys[:traj_len], xs[:traj_len], us[:traj_len]


def solve_admm2(slsys, y0, x0, traj_len, ε=1e-4, ρ=0.1, max_iter=10):
    """
    Solve the two minimizations alternatively:

    minimize_v ∑ₜ yₜQₜyₜ + wₖₜᵀ(E xₖₜ - vₜ) + 0.5 ρ|E xₖₜ - vₜ|²
    s.t.          yₜ₊₁ = Ay yₜ + Bv vₜ

    minimize_u ∑ₜ uₜRuₜ + wₖₜᵀ(E xₜ - vₖₜ) + 0.5 ρ|E xₜ - vₖₜ|²
    s.t.          xₜ₊₁ = Ax xₜ₊₁ + Bu uₜ

    """
    warnings.warn("solve_admm2: This implementation does not provide good result. Use solve_admm instead")
    E = slsys.E
    ys, xs0, us0 = solve_seq(slsys, y0, x0, traj_len)
    vs0 = [E.dot(xt) for xt in xs0[1:]]
    ws0 = [np.zeros_like(vt) for vt in vs0]
    return admm((partial(slsys.prox_f_v, y0, x0),
                 partial(slsys.prox_g_u, y0, x0)),
                vs0, us0, ws0, partial(slsys.constraint_fn, y0, x0), ρ)


_SeparableLinearSystem = namedtuple('_SeparableLinearSystem',
                                    "Qy R Ay Bv QyT E Ax Bu T".split(" "))


def randps(*dim, rng=np.random):
    """
    Random positive definite matrix?
    """
    Msqrt = rng.rand(*dim)
    return Msqrt.T.dot(Msqrt)


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
        return cls(Qy  = randps(yD, yD, rng=rng),
                   R   = randps(uD, uD, rng=rng),
                   Ay  = rng.rand(yD, yD),
                   Bv  = rng.rand(yD, uD),
                   QyT = randps(yD, yD, rng=rng),
                   E   = rng.rand(vD, xD),
                   Ax  = rng.rand(xD, xD),
                   Bu  = rng.rand(xD, uD),
                   T   = T)

    def forward_x(self, x0, us):
        Ax = self.Ax
        Bu = self.Bu
        xs = [x0]
        for t, ut in enumerate(us):
            xs.append(Ax.dot(xs[-1]) + Bu.dot(ut))
        return xs

    def forward_y(self, y0, vs):
        Ay = self.Ay
        Bv = self.Bv
        ys = [y0]
        for t, vt in enumerate(vs):
            ys.append(Ay.dot(ys[-1]) + Bv.dot(vt))
        return ys

    def forward(self, y0, x0, us):
        # Generate forward trajectory
        E  = self.E

        xs = self.forward_x(x0, us)
        vs = [E.dot(xt) for xt in xs]
        ys = self.forward_y(y0, vs)

        return ys, xs


    def costs(self, ys, xs, us):
        Q = lambda t: self.QyT if t >= self.T else self.Qy
        R = self.R
        return [yt.T.dot(Q(t)).dot(yt) + (0
                                          if ut is None
                                          else ut.T.dot(R).dot(ut))
                for t, (yt, ut) in enumerate(zip_longest(ys, us))]

    def prox_g_u(self, y0, x0, vs, _, ws, ρ):
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
        x_sys   = LinearSystem(Axh,                    Buh,
                               Qxs,                    np.zeros(Qxs[0].shape[0]),
                               R,                      np.zeros(R.shape[0]),
                               np.zeros_like(Qxs[-1]), np.zeros(Qxs[-1].shape[0]),
                               T)
        xhs_new, us_new, usmin = x_sys.solve(x0h, len(vs), return_min=True)
        assert len(xhs_new) == len(vs) + 1
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
        assert len(xs) == len(us) + 1
        Rsv = [0.5 * ρ * np.eye(E.shape[0])
               for _ in xs]
        zsv = [0.5 * ρ * ( -E.dot(x) - w/ρ )
               for x, w in zip(xs[1:], ws)]

        # v0 is fixed because x0, v_{1:T} is unknown
        v0 = E.dot(x0)
        y1 = Ay.dot(y0) + Bv.dot(v0)

        y_sys  = LinearSystem(Ay, Bv,
                              Qy, np.zeros(Qy.shape[0]),
                              Rsv, zsv,
                              QyT, np.zeros(QyT.shape[0]), T)
        ys_new, vs_new, vsmin = y_sys.solve(y1, len(us), return_min=True)
        assert len(vs_new) == len(us)
        assert len(ys_new) == len(vs_new) + 1
        return np.vstack(vs_new)

    def constraint_fn(self, y0, x0, vs_us):
        vD = self.Bv.shape[0]
        vs = vs_us[:, :vD]
        us = vs_us[:, vD:]

        E  = self.E
        assert len(vs) == len(us)
        ulen = len(us)

        xs = self.forward_x(x0, us)
        assert len(xs) == ulen + 1
        # Need to vstack it because it should support the addition operator
        ws = np.vstack(([E.dot(xt) - vt for xt, vt in zip(xs[1:], vs)]))
        assert ws.shape == (len(vs), vs[0].shape[0])
        return ws


def plotables(ys, xs, us, linsys, traj_len):
    costs = linsys.costs(ys, xs, us)[:traj_len]
    return [("pos", np.array([y[0] for y in ys[:traj_len]])),
            ("vel", np.array([x[0] for x in xs[:traj_len]])),
            ("ctrl", np.array(us[:traj_len])),
            ("cost", costs)]


def quadrotor_square_example():
    _, y0, Ay, Bv, Qy, _, _, _, QyT, _, _ = quadrotor_linear_system()
    _, x0, Ax, Bu, _, _, R, _, _, _, _ = quadrotor_linear_system()
    E = np.hstack((np.eye(Bv.shape[1]),
                   np.zeros((Bv.shape[1], Ax.shape[1] - Bv.shape[1]))))
    return plotables, y0, x0, Qy, R, Ay, Bv, QyT, E, Ax, Bu, 100


def quadrotor_as_separable(m  = 1,
                           r0 = 1,
                           Ay = [[1.]],
                           Bv = [[1.]],
                           E  = [[1.]],
                           Qy = [[1]],
                           Ax = [[1.]],
                           y0 = [-1],
                           x0 = [0],
                           T  = 100):
    Bu=[[1/m]]
    R=[[r0]]
    QyT = np.array(Qy)*100
    return [plotables] + list(map(np.asarray, (y0, x0, Qy, R, Ay, Bv, QyT, E, Ax, Bu, T)))


def plot_separable_sys_results(example=quadrotor_square_example, traj_len=30):
    plotables, y0, x0, *sepsys = example()
    fig = None
    slsys = SeparableLinearSystem(*sepsys)
    solvers = (solve_full, solve_seq, solve_admm)
    labels = map(attrgetter('__name__'), solvers)
    short_labels = diff_substr(labels)
    eff_traj_len = min(slsys.T, traj_len)
    for solver, label in zip(solvers, short_labels):
        ys_full, xs_full, us_full = solver(slsys, y0, x0, eff_traj_len)
        fig = plot_solution(np.arange(eff_traj_len),
                            plotables(ys_full, xs_full, us_full, slsys, eff_traj_len),
                            axes= None if fig is None else fig.axes,
                            plot_fn=partial(Axes.plot, label=label))
    if fig is not None:
        fig.show()
        plt.show()

if __name__ == '__main__':
    plot_separable_sys_results(example=quadrotor_as_separable)

