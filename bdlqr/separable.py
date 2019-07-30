from functools import partial
from collections import namedtuple
from operator import attrgetter
from itertools import starmap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from bdlqr.full import LinearSystem, quadrotor_linear_system, plot_solution
from bdlqr.diff_substr import diff_substr


def solve_seq(slsys, y0, x0):
    """
    minimize_v ∑ₜ yₜQₜyₜ
    s.t.          yₜ₊₁ = Ay yₜ + Bv vₜ

    minimize_u ∑ₜ uₜRuₜ + |E xₜ - vₜ|₂²
    s.t.          xₜ₊₁ = Ax xₜ₊₁ + Bu uₜ
    """
    Qy  = slsys.Qy
    R    = slsys.R
    Ay  = slsys.Ay
    Bv   = slsys.Bv
    QyT = slsys.QyT
    E    = slsys.E
    Ax   = slsys.Ax
    Bu   = slsys.Bu
    T    = slsys.T
    y_sys = LinearSystem(Ay, Bv, Qy, R, QyT, T)
    ys, vs = y_sys.solve(y0)

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
    x_sys = LinearSystem(Axh, Buh, Qsx[:-1], R, Qsx[-1], T)
    xhs, us = x_sys.solve(x0h)
    xs = [xh[:-1] for xh in xhs]
    assert xs[0].shape[0] == x0.shape[0]
    assert ys[0].shape[0] == y0.shape[0]
    assert us[0].shape[0] == Bu.shape[1]
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


def solve_full(slsys, y0, x0):
    A, B, Q, R, QT, T, X0 = joint_linear_system(slsys, y0, x0)
    Xs, us = LinearSystem(A, B, Q, R, QT, T).solve(X0)
    ys = [X[:y0.shape[0]] for X in Xs]
    xs = [X[y0.shape[0]:] for X in Xs]
    return ys, xs, us


def solve_admm(slsys, y0, x0, ε=1e-2, ρ=1, max_iter=10):
    """
    Solve the two minimizations alternatively:

    minimize_v ∑ₜ yₜQₜyₜ + wₖₜᵀ(E xₖₜ - vₜ) + 0.5 ρ|E xₖₜ - vₜ|²
    s.t.          yₜ₊₁ = Ay yₜ + Bv vₜ

    minimize_u ∑ₜ uₜRuₜ + wₖₜᵀ(E xₜ - vₖₜ) + 0.5 ρ|E xₜ - vₖₜ|²
    s.t.          xₜ₊₁ = Ax xₜ₊₁ + Bu uₜ

    """
    Qy  = slsys.Qy
    R    = slsys.R
    Ay  = slsys.Ay
    Bv   = slsys.Bv
    QyT = slsys.QyT
    E    = slsys.E
    Ax   = slsys.Ax
    Bu   = slsys.Bu
    T    = slsys.T
    ys, xs, us = solve_seq(slsys, y0, x0)
    vs = [E.dot(x) for x in xs]
    ws = [np.zeros(vs[0].shape[0]) for _ in range(T)]

    for k in range(max_iter):
        #
        # minimize_v ∑ₜ yₜQₜyₜ + wₖₜᵀ(E xₖₜ - vₜ) + 0.5 ρ|E xₖₜ - vₜ|²
        # s.t.          yₜ₊₁ = Ay yₜ + Bv vₜ
        # Let ṽₜ = [ vₜ]
        #          [ 1 ]
        # Rv = 0.5 ρ [I, - E xₖₜ + wₖₜ/ρ]ᵀ [ I, - E xₖₜ + wₖₜ/ρ  ]
        # Bvh = [Bv, 0]
        sqrtRs = [np.hstack(( np.eye(vs[0].shape[0]), (- E.dot(x) + w/ρ).reshape(-1,1) ))
                  for x, w in zip(xs, ws)]
        Rsvh = [0.5 * ρ * sqrtR.T.dot(sqrtR)
               for sqrtR in sqrtRs]
        Bvh = np.hstack((Bv, np.zeros((Bv.shape[0], 1))))
        y_sys  = LinearSystem(Ay, Bvh, Qy, Rsvh, QyT, T)
        ys_new, vhs_new = y_sys.solve(y0)
        vs_new = [vh_new[:-1] for vh_new in vhs_new]

        # Reformulate the affine to a linear system by making the state a
        # homogeneous vector
        # minimize_u ∑ₜ uₜRuₜ + wₖₜᵀ(E xₜ - vₖₜ) + 0.5 ρ|E xₜ - vₖₜ|²
        # s.t.          xₜ₊₁ = Ax xₜ₊₁ + Bu uₜ
        # where Qxh = [  Eᵀ           ] [ E  -vₖₜ - wₖₜ/ρ]
        #             [ -vₖₜᵀ - wₖₜᵀ/ρ]
        #  and  Axh = [ A   0 ]
        #             [ 0   1 ]
        #  and  Buh = [ Bu ]
        #             [ 0  ]
        #  and xhₜ =  [ xₜ ]
        #             [ 1  ]
        x0h    = np.hstack((x0, 1))
        E_vs    = [np.hstack((E, (-v_new - w/ρ).reshape(-1,1)))
                   for v_new, w in zip(vs_new, ws)]
        Qxs    = [E_vt.T.dot(E_vt)    for E_vt in E_vs]
        Axh     = np.eye(x0h.shape[0])
        Axh[:-1, :-1] = Ax
        Buh     = np.vstack((Bu, 0))
        x_sys   = LinearSystem(Axh, Buh, Qxs[:-1], R, Qxs[-1], T)
        xhs_new, us_new = x_sys.solve(x0h)
        xs_new = [xh[:-1] for xh in xhs_new]

        change = sum(np.linalg.norm(u - u_new)
                     for u, u_new in zip(us, us_new))

        ys = ys_new
        vs = vs_new
        xs = xs_new
        us = us_new

        if change < ε:
            break

    return ys, xs, us


_SeparableLinearSystem = namedtuple('_SeparableLinearSystem',
                                    "Qy R Ay Bv QyT E Ax Bu T".split(" "))


class SeparableLinearSystem(_SeparableLinearSystem):
    """
    Represents a linear system problem of the form:

    minimize_u  ∑ₜ yₜQₜyₜ + uₜRuₜ + y_T Q_T y_T
    s.t.          yₜ₊₁ = Ay yₜ + Bv vₜ
                    vₜ = E xₜ
                  xₜ₊₁ = Ax xₜ₊₁ + Bu uₜ
    """
    def cost(self, x_t, u_t, t):
        Q = self.QyT if t >= self.T else self.Qy
        R = self.R
        return x_t.T.dot(Q).dot(x_t) + (0
                                        if t >= self.T
                                        else u_t.T.dot(R).dot(u_t))


def quadrotor_square_example():
    _, y0, Ay, Bv, Qy, _, QyT, T = quadrotor_linear_system()
    _, x0, Ax, Bu, _, R, _, T = quadrotor_linear_system()
    E = np.hstack((np.eye(Bv.shape[1]),
                   np.zeros((Bv.shape[1], Ax.shape[1] - Bv.shape[1]))))
    def plotables(ys, xs, us, linsys):
        costs = list(starmap(linsys.cost, zip(xs[1:], us, range(1, len(us)+1))))
        return [("y[0]", np.array([x[0] for x in ys[1:]])),
                ("y[1]", np.array([x[0] for x in ys[1:]])),
                ("u[0]", np.array(us)),
                ("cost", costs)]

    return plotables, y0, x0, Qy, R, Ay, Bv, QyT, E, Ax, Bu, T


def quadrotor_as_separable(m=1,
                           r0=10,
                           Ay=[[1.]],
                           Bv=[[1.]],
                           E=[[1.]],
                           Qy=[[1]],
                           Ax=[[1.]],
                           T=30):
    Bu=[[1/m]]
    R=[[r0]]
    QyT = [[100]]
    y0 = np.array([-1])
    x0 = np.array([0])
    def plotables(ys, xs, us, linsys):
        costs = list(starmap(linsys.cost, zip(xs[1:], us, range(1, len(us)+1))))
        return [("pos", np.array([x[0] for x in ys[1:]])),
                ("vel", np.array([x[0] for x in xs[1:]])),
                ("ctrl", np.array(us)),
                ("cost", costs)]
    return [plotables] + list(map(np.array, (y0, x0, Qy, R, Ay, Bv, QyT, E, Ax, Bu, T)))


def plot_separable_sys_results(example=quadrotor_square_example):
    plotables, y0, x0, *sepsys = example()
    fig = None
    slsys = SeparableLinearSystem(*sepsys)
    solvers = (solve_full, solve_seq, solve_admm)
    labels = map(attrgetter('__name__'), solvers)
    short_labels = diff_substr(labels)
    for solver, label in zip(solvers, short_labels):
        ys_full, xs_full, us_full = solver(slsys, y0, x0)
        fig = plot_solution(np.arange(1, slsys.T+1),
                            plotables(ys_full, xs_full, us_full, slsys),
                            axes= None if fig is None else fig.axes,
                            plot_fn=partial(Axes.plot, label=label))
    if fig is not None:
        fig.show()
        plt.show()

if __name__ == '__main__':
    plot_separable_sys_results()
    plot_separable_sys_results(example=quadrotor_as_separable)

