from functools import partial
from collections import namedtuple
from operator import attrgetter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from bdlqr.full import LinearSystem, quadrotor_linear_system, plot_solution
from bdlqr.diff_substr import diff_substr

def solve_seq(self, y_0, x_0):
    """
    minimize_v ∑ₜ yₜQₜyₜ
    s.t.          yₜ₊₁ = A_y yₜ + Bᵥ vₜ

    minimize_u ∑ₜ uₜRuₜ + |E xₜ - vₜ|₂
    s.t.          xₜ₊₁ = Aₓ xₜ₊₁ + Bᵤ uₜ
    """
    Q_y  = self.Q_y
    R    = self.R
    A_y  = self.A_y
    Bᵥ   = self.Bᵥ
    Q_yT = self.Q_yT
    E    = self.E
    Aₓ   = self.Aₓ
    Bᵤ   = self.Bᵤ
    T    = self.T
    y_sys = LinearSystem(A_y, Bᵥ, Q_y, R, Q_yT, T)
    ys, vs = y_sys.solve(y_0)

    # Reformulate the affine to a linear system by making the state a
    # homogeneous vector
    # minimize_u ∑ₜ uₜRuₜ + xₕₜ Qₓₕ xₕₜ
    # s.t.          xₕₜ₊₁ = Aₓₕ xₕₜ₊₁ + Bᵤₕ uₜ
    # where Qₓₕ = [  Eᵀ ] [ E  -vₜ]
    #             [ -vᵀ ]
    #  and  Aₓₕ = [ A   0 ]
    #             [ 0   1 ]
    #  and  Bᵤₕ = [ Bᵤ ]
    #             [ 0  ]
    #  and xₕₜ =  [ xₜ ]
    #             [ 1  ]
    x_0h = np.hstack((x_0, 1))
    E_vs = [np.hstack((E, -vₜ.reshape(-1, 1)))
            for vₜ in vs]
    Qsₓ = [E_vt.T.dot(E_vt)
            for E_vt in E_vs]
    Aₓₕ = np.eye(x_0h.shape[0])
    Aₓₕ[:-1, :-1] = Aₓ
    Bᵤₕ = np.vstack((Bᵤ, 0))
    x_sys = LinearSystem(Aₓₕ, Bᵤₕ, Qsₓ[:-1], R, Qsₓ[-1], T)
    xhs, us = x_sys.solve(x_0h)
    xs = [xh[:-1] for xh in xhs]
    assert xs[0].shape[0] == x_0.shape[0]
    assert ys[0].shape[0] == y_0.shape[0]
    assert us[0].shape[0] == Bᵤ.shape[1]
    return ys, xs, us


def joint_linear_system(slsys, y_0, x_0):
    self = slsys
    Q_y  = self.Q_y
    R    = self.R
    A_y  = self.A_y
    Bᵥ   = self.Bᵥ
    Q_yT = self.Q_yT
    E    = self.E
    Aₓ   = self.Aₓ
    Bᵤ   = self.Bᵤ
    T    = self.T
    # Q = [Q_y  0 ]
    #     [ 0   0 ]
    # Xₜ = [yₜ]
    #      [xₜ]
    # A = [A_y BᵥE ]
    #     [ 0  Aₓ  ]
    #
    # B = [ 0  ]
    #     [ Bᵤ ]
    X_0 = np.hstack((y_0, x_0))
    Q = np.vstack((np.hstack((Q_y, np.zeros((Q_y.shape[0], x_0.shape[0])))),
                   np.zeros((x_0.shape[0], X_0.shape[0]))))
    QT = np.vstack((np.hstack((Q_yT, np.zeros((Q_yT.shape[0], x_0.shape[0])))),
                    np.zeros((x_0.shape[0], X_0.shape[0]))))
    A = np.vstack((np.hstack((A_y, Bᵥ.dot(E))),
                    np.hstack((np.zeros((Aₓ.shape[0], A_y.shape[1])), Aₓ))))
    B = np.vstack((np.zeros((y_0.shape[0], Bᵤ.shape[1])),
                   Bᵤ))
    return A, B, Q, R, QT, T, X_0


def solve_full(self, y_0, x_0):
    A, B, Q, R, QT, T, X_0 = joint_linear_system(self, y_0, x_0)
    Xs, us = LinearSystem(A, B, Q, R, QT, T).solve(X_0)
    ys = [X[:y_0.shape[0]] for X in Xs]
    xs = [X[y_0.shape[0]:] for X in Xs]
    return ys, xs, us


def solve_admm(self, y_0, x_0, ε=1e-2, ρ=1, max_iter=10):
    """
    Solve the two minimizations alternatively:

    minimize_v ∑ₜ yₜQₜyₜ + wₖₜᵀ(E xₖₜ - vₜ) + 0.5 ρ|E xₖₜ - vₜ|²
    s.t.          yₜ₊₁ = A_y yₜ + Bᵥ vₜ

    minimize_u ∑ₜ uₜRuₜ + wₖₜᵀ(E xₜ - vₖₜ) + 0.5 ρ|E xₜ - vₖₜ|²
    s.t.          xₜ₊₁ = Aₓ xₜ₊₁ + Bᵤ uₜ

    """
    Q_y  = self.Q_y
    R    = self.R
    A_y  = self.A_y
    Bᵥ   = self.Bᵥ
    Q_yT = self.Q_yT
    E    = self.E
    Aₓ   = self.Aₓ
    Bᵤ   = self.Bᵤ
    T    = self.T
    ysₖ, xsₖ, usₖ = solve_seq(self, y_0, x_0)
    vsₖ = [E.dot(xₖₜ) for xₖₜ in xsₖ]
    wsₖ = [np.zeros(vsₖ[0].shape[0]) for _ in range(T)]

    change = 1 + ε
    k = 0
    while change > ε and k < max_iter:
        #
        # minimize_v ∑ₜ yₜQₜyₜ + wₖₜᵀ(E xₖₜ - vₜ) + 0.5 ρ|E xₖₜ - vₜ|²
        # s.t.          yₜ₊₁ = A_y yₜ + Bᵥ vₜ
        # Let ṽₜ = [ vₜ]
        #          [ 1 ]
        # Rᵥ = 0.5 ρ [I, - E xₖₜ + wₖₜ/ρ]ᵀ [ I, - E xₖₜ + wₖₜ/ρ  ]
        # Bᵥₕ = [Bᵥ, 0]
        sqrtRs = [np.hstack(( np.eye(vsₖ[0].shape[0]), (- E.dot(xₖₜ) + wₖₜ/ρ).reshape(-1,1) ))
                  for xₖₜ, wₖₜ in zip(xsₖ, wsₖ)]
        Rsᵥₕ = [0.5 * ρ * sqrtR.T.dot(sqrtR)
               for sqrtR in sqrtRs]
        Bᵥₕ = np.hstack((Bᵥ, np.zeros((Bᵥ.shape[0], 1))))
        y_sys  = LinearSystem(A_y, Bᵥₕ, Q_y, Rsᵥₕ, Q_yT, T)
        ysₖₙ, vhsₖₙ = y_sys.solve(y_0)
        vsₖₙ = [vhₖₙ[:-1] for vhₖₙ in vhsₖₙ]

        # Reformulate the affine to a linear system by making the state a
        # homogeneous vector
        # minimize_u ∑ₜ uₜRuₜ + wₖₜᵀ(E xₜ - vₖₜ) + 0.5 ρ|E xₜ - vₖₜ|²
        # s.t.          xₜ₊₁ = Aₓ xₜ₊₁ + Bᵤ uₜ
        # where Qₓₕ = [  Eᵀ           ] [ E  -vₖₜ - wₖₜ/ρ]
        #             [ -vₖₜᵀ - wₖₜᵀ/ρ]
        #  and  Aₓₕ = [ A   0 ]
        #             [ 0   1 ]
        #  and  Bᵤₕ = [ Bᵤ ]
        #             [ 0  ]
        #  and xₕₜ =  [ xₜ ]
        #             [ 1  ]
        x_0h    = np.hstack((x_0, 1))
        E_vs    = [np.hstack((E, (-vₖₙₜ - wₖₜ/ρ).reshape(-1,1)))
                   for vₖₙₜ, wₖₜ in zip(vsₖₙ, wsₖ)]
        Qₓₛ    = [E_vt.T.dot(E_vt)    for E_vt in E_vs]
        Aₓₕ     = np.eye(x_0h.shape[0])
        Aₓₕ[:-1, :-1] = Aₓ
        Bᵤₕ     = np.vstack((Bᵤ, 0))
        x_sys   = LinearSystem(Aₓₕ, Bᵤₕ, Qₓₛ[:-1], R, Qₓₛ[-1], T)
        xhsₖₙ, usₖₙ = x_sys.solve(x_0h)
        xsₖₙ = [xh[:-1] for xh in xhsₖₙ]

        # Loop variables
        change = sum(np.linalg.norm(uₖₜ - uₖₙₜ)
                     for uₖₜ, uₖₙₜ in zip(usₖ, usₖₙ))
        k += 1

        # Loop state
        ysₖ = ysₖₙ
        vsₖ = vsₖₙ
        xsₖ = xsₖₙ
        usₖ = usₖₙ

    return ysₖ, xsₖ, usₖ



_SeparableLinearSystem = namedtuple('_SeparableLinearSystem',
                                    "Q_y R A_y Bᵥ Q_yT E Aₓ Bᵤ T".split(" "))
class SeparableLinearSystem(_SeparableLinearSystem):
    """
    Represents a linear system problem of the form:

    minimize_u  ∑ₜ yₜQₜyₜ + uₜRuₜ + y_T Q_T y_T
    s.t.          yₜ₊₁ = A_y yₜ + Bᵥ vₜ
                    vₜ = E xₜ
                  xₜ₊₁ = Aₓ xₜ₊₁ + Bᵤ uₜ
    """
    def cost(self, x_t, u_t, t):
        Q = self.Q_yT if t >= self.T else self.Q_y
        R = self.R
        return x_t.T.dot(Q).dot(x_t) + (0
                                        if t >= self.T
                                        else u_t.T.dot(R).dot(u_t))


def separable_linear_system_example():
    A_y, Bᵥ, Q_y, _, Q_yT, T = quadrotor_linear_system()
    Aₓ, Bᵤ, _, R, _, T = quadrotor_linear_system()
    E = np.hstack((np.eye(Bᵥ.shape[1]),
                   np.zeros((Bᵥ.shape[1], Aₓ.shape[1] - Bᵥ.shape[1]))))
    y_0 = np.array([-1, 0])
    x_0 = np.array([-1, 0])
    return Q_y, R, A_y, Bᵥ, Q_yT, E, Aₓ, Bᵤ, T, y_0, x_0


if __name__ == '__main__':
    sepsys_X0 = separable_linear_system_example()
    y_0, x_0 = sepsys_X0[-2:]
    sepsys = sepsys_X0[:-2]
    fig = None
    slsys = SeparableLinearSystem(*sepsys)
    solvers = (solve_full, solve_seq, solve_admm)
    labels = map(attrgetter('__name__'), solvers)
    short_labels = diff_substr(labels)
    for solver, label in zip(solvers, short_labels):
        ys_full, xs_full, us_full = solver(slsys, y_0, x_0)
        fig = plot_solution(slsys.cost, ys_full, us_full, np.arange(1, slsys.T+1),
                            axes= None if fig is None else fig.axes,
                            plot_fn=partial(Axes.plot, label=label))
    if fig is not None:
        fig.show()
        plt.show()



