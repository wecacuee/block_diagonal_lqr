from collections import deque
from functools import partial
from itertools import zip_longest

import numpy as np
from numpy.linalg import norm
from bdlqr.separable import SeparableLinearSystem, joint_linear_system, solve_seq
from bdlqr.linalg import ScalarQuadFunc
from bdlqr.lqr import affine_backpropagation

from scipy.optimize import fmin


class MaskEnvDynamics:
    def __init__(self, slsys):
        self._slsys = slsys

    def Ay(self):
        raise NotImplementedError("Env dynamics not allowed")

    def Bv(self):
        raise NotImplementedError("Env dynamics not allowed")

    def __getattr__(self, a):
        return getattr(self._slsys, a)


def independent_cost_to_go(slsys, max_iter=100):
    """
    V(y₀) = arg min_v ∑ₜ c(yₜ)
            s.t.  yₜ₊₁ = Ay yₜ + Bv vₜ   ∀ t

    Let Vₜ(yₜ) = yₜᵀ Pₜ yₜ + 2 oₜᵀyₜ
    return ScalarQuadFunc(Pₜ, oₜ, 0)

    >>> seed = np.random.randint(10000)
    >>> rng = np.random.RandomState(seed)
    >>> slsys = SeparableLinearSystem.random(T=3, rng=rng)
    >>> yD = slsys.Ay.shape[-1]
    >>> xD = slsys.Ax.shape[-1]
    >>> uD = slsys.Bu.shape[-1]
    >>> y0 = rng.rand(yD)
    >>> x0 = rng.rand(xD)
    >>> us = [rng.rand(uD) for _ in range(slsys.T)]
    >>> ys, xs = slsys.forward(y0, x0, us)
    >>> V = independent_cost_to_go(slsys)
    >>> (V(y0) <= sum(yt.T.dot(slsys.Qy).dot(yt) for yt in ys)).all()
    True
    """
    Qy   = slsys.Qy
    Ay   = slsys.Ay
    Bv   = slsys.Bv
    QyT  = slsys.QyT
    T    = slsys.T

    yD  = Ay.shape[-1]
    vD  = Bv.shape[-1]
    sy  = np.zeros(yD)
    Rv  = np.zeros((vD, vD))
    zv  = np.zeros(vD)
    oyT = np.zeros(yD)

    # Loop back from T
    iterations = min(max_iter, T)
    P_new, o_new = QyT, oyT
    for _ in range(iterations):
        P_new, o_new, K, k = affine_backpropagation(
            Qy, sy, Rv, zv, Ay, Bv, P_new, o_new)
    return ScalarQuadFunc(P_new, o_new, np.array(0))


def proximal_cost_to_go(slsys, GV, ṽks, ρ):
    """
    GV(y₀) := arg min_v ∑ₜ yₜᵀQyₜ
            s.t.  yₜ₊₁ = Ay yₜ + Bv vₜ   ∀ t

    Vₚᵣₒₓ(y₁, ṽs₁) = minᵥ ∑ⁿₜ₌₁ yₜᵀQyₜ + 0.5 ρ |vₜ - ṽₜ| + GV(yₖ₊₁)

    Let Vₜ(yₜ) = yₜᵀ Pₜ(ṽs₁) yₜ + 2 oₜᵀyₜ
    return ScalarQuadFunc(Pₜ, oₜ, 0)

    >>> seed = np.random.randint(10000)
    >>> rng = np.random.RandomState(seed)
    >>> slsys = SeparableLinearSystem.random(T=3, rng=rng)
    >>> yD = slsys.Ay.shape[-1]
    >>> xD = slsys.Ax.shape[-1]
    >>> uD = slsys.Bu.shape[-1]
    >>> y0 = rng.rand(yD)
    >>> x0 = rng.rand(xD)
    >>> us = [rng.rand(uD) for _ in range(slsys.T)]
    >>> ys, xs = slsys.forward(y0, x0, us)
    >>> vD = slsys.Bv.shape[-1]
    >>> vks = [rng.rand(vD)]
    >>> ρ = 1
    >>> GV = independent_cost_to_go(slsys)
    >>> V = proximal_cost_to_go(slsys, GV, vks, ρ)
    >>> (V(y0) <= sum(yt.T.dot(slsys.Qy).dot(yt) +
    ...              (0 if vkt is None else 0.5 * ρ * norm(vkt - slsys.E.dot(xt)))
    ...              for yt, xt, vkt in zip_longest(ys, xs, vks))).all()
    True
    """
    n = len(ṽks)
    Qy   = slsys.Qy
    Ay   = slsys.Ay
    Bv   = slsys.Bv
    QyT  = slsys.QyT
    T    = slsys.T

    yD  = Ay.shape[-1]
    vD  = Bv.shape[-1]
    vhD = vD + 1
    # V(y, ṽ) = arg min_v ∑_ṽ 0.5 ρ |ṽₜ - vₜ]_2^2 + ∑ₜ y Q y
    #            yₜ₊₁   = Ay yₜ   + Bv vₜ
    # V(y, ṽ) = arg min_v ∑ y Q y + 0.5 ρ [vₜᵀ, 1] [    1, -ṽₜ]  [vₜ ]
    #                                              [-ṽₜᵀ,  ṽ^2]  [   1]
    #            yₜ₊₁   = Ay yₜ   + [Bv, 0] [vₜ]
    #                                       [ 1]
    sy  = np.zeros(yD)
    zv  = np.zeros(vhD)
    oyT = np.zeros(yD)
    Bvh = np.hstack((Bv, np.zeros((yD, 1))))
    ṽks_col_vec = [ṽt[:, None] for ṽt in ṽks]
    Rsh = [0.5 * ρ * np.vstack((np.hstack((np.eye(vD),         - ṽt)),
                                np.hstack((    - ṽt.T, ṽt.T.dot(ṽt)))))
           for ṽt in ṽks_col_vec]
    P_vs = deque([GV.Q])
    o_vs = deque([GV.l])
    K_vs = deque([])
    k_vs = deque([])
    for Rht in reversed(Rsh):
        Pt, ot, Kt, kt = affine_backpropagation(
            Qy, sy, Rht, zv, Ay, Bvh, P_vs[-1], o_vs[-1])
        P_vs.appendleft(Pt)
        o_vs.appendleft(ot)
        K_vs.appendleft(Kt)
        k_vs.appendleft(kt)
    return ScalarQuadFunc(P_vs[0], o_vs[0], np.array(0))


def greedy_proximal_q_function(slsys, GV, y, ṽs, v, ρ):
    """
    Qₚᵣₒₓ(y₀, ṽs, v₀) := c(y₀) + 0.5 ρ |v₀ - ṽ₀|
                        + minᵥ ∑ᵏₜ₌₁ c(yₜ) + 0.5 ρ |vₜ - ṽₜ| + GV(yₖ₊₁)
            s.t.  yₜ₊₁ = Ay yₜ + Bv vₜ   ∀ t
            where k = len(ṽs)

    return Qₚᵣₒₓ(y₀, ṽs, v)

    """
    Qy   = slsys.Qy
    Ay   = slsys.Ay
    Bv   = slsys.Bv
    y1 = Ay.dot(y) + Bv.dot(v)
    # GV = proximal_cost_to_go(slsys, independent_cost_to_go(slsys), ṽs[1:], ρ)
    return (y.T.dot(Qy).dot(y) + 0.5 * ρ * norm(v - ṽs[0]) + GV(y1))


def argmin_greedy_proximal_q_function(slsys, GV, y, ṽs, ρ):
    """
    return arg min_v Qₚᵣₒₓ(y₀, ṽs, v)
    """
    Qy   = slsys.Qy
    Ay   = slsys.Ay
    Bv   = slsys.Bv

    yD  = Ay.shape[-1]
    vD  = Bv.shape[-1]
    vhD = vD + 1
    sy  = np.zeros(yD)
    ṽks_col_vec = [ṽt[:, None] for ṽt in ṽks]
    Rsh = [0.5 * ρ * np.vstack((np.hstack((np.eye(vD),         - ṽt)),
                                np.hstack((    - ṽt.T, ṽt.T.dot(ṽt)))))
           for ṽt in ṽks_col_vec]
    zv  = np.zeros(vhD)
    Bvh = np.hstack((Bv, np.zeros((yD, 1))))
    P_new, o_new, K, k = affine_backpropagation(
        Qy, sy, Rht, zv, Ay, Bvh, GV.Q, GV.l)
    return K.dot(y) + k


def approx_vtilde_dynamics(slsys, yt, vt, ws):
    """
    return argmin_u uₜᵀRₜuₜ + 0.5*ρ*|Exₜ - vₜ + wₜ/ρ|
                 s.t. xₜ₊₁ = Ax xₜ + Bu uₜ
    """
    x0 = np.linalg.solve(E, vt)[0]
    u0 = slsys.prox_g_u(None, x0, [vt], None, ws[:1])
    Ax = slsys.Ax
    Bu = slsys.Bu
    return E.dot(Ax.dot(x0) + Bu.dot(u0)) + ws[0]/ρ


def ground_true_env_dynamics(slsys, yt, vt):
    Ay = slsys.Ay
    Bv = slsys.Bv
    return Ay.dot(yt) + Bv.dot(vt)


def approx_env_dynamics(masked_slsys, Vprox, yt, xt, wt, vt, ρ):
    """
    yₜ₊₁ = arg min_y | Vprox(yₜ, ṽₜ) - yₜᵀQyₜ - 0.5 * ρ |ṽₜ - vₜ|₂² -  Vprox(y) |
    return yₜ₊₁
    """
    Q = masked_slsys.Q
    E = masked_slsys.E
    ṽt = E.dot(xt) + wt/ρ
    cost_t = yt.dot(Q).dot(yt) + 0.5 * ρ * (ṽt-vt).T.dot(ṽt-vt)
    target_value = Vprox(yt, ṽt) - cost_t
    cost = lambda y: target_value - Vprox(y, approx_vtilde_dynamics(slsys, y, vt, [wt], ρ))
    return fmin(cost, yt)


def planning_from_learned_q_function(masked_slsys1, Q1, argmin_Q1,
                                     masked_slsys2, y0, x0, vs, us, wks,
                                     env_dynamics, vtilde_dynamics):
    slsys2 = masked_slsys2
    T = slsys2.T
    def prox_env(_, uks, wks, ρ):
        """
        return arg min_{vs} h(xks + wks / ρ)
        """
        Ax = slsys2.Ax
        Bu = slsys2.Bu
        E = slsys2.E
        xks, uks = slsys2.forward_x(x0, us)
        ṽks = [E.dot(xt) + wt/ρ
               for xt, wt in zip(xks, wks)]
        # Q1 can only return one step minimization
        # TODO: We have two options to use ground truth dynamics or approximate
        # dynamics
        yt = y0
        vs = []
        for t in range(T):
            vt = Q1_argmin(yt, ṽks[t:], ρ)
            ytp1 = env_dynamics(yt, vt)
            ṽks[t+1] = vtilde_dynamics(y, vt)
            yt = ytp1
            vs.append(vt)

        return vs

    prox_robot = slsys2.prox_g_u
    const_fn = partial(slsys2.constraint_fn, y0, x0)
    wks = admm_wk(vks, xks, const_fn, None)
    vks_admm, uks_admm, wks_admm = admm((prox_env, prox_robot), vks, uks, wks,
                                        const_fn, ρ)
    return vks_admm, uks_admm


def quadrotor_as_separable(Ay  = [[1.]],
                           Qy  = [[1]],
                           Bv  = [[1.]],
                           r01 = 1,
                           m1  = 1,
                           E1  = [[1.]],
                           Ax1 = [[1.]],
                           r02 = 2,
                           m2  = 2,
                           E2  = [[2.]],
                           Ax2 = [[2.]],
                           y0  = [-1],
                           x0  = [0],
                           T   = np.Inf):
    Bu1=[[1/m1]]
    R1=[[r01]]
    Bu2=[[1/m2]]
    R2=[[r02]]
    QyT = np.array(Qy)*100
    return (SeparableLinearSystem(*map(np.array, (Qy, R1, Ay, Bv, QyT, E1, Ax1, Bu1, T))),
            SeparableLinearSystem(*map(np.array, (Qy, R2, Ay, Bv, QyT, E2, Ax2, Bu2, T))))


def test_transfer_separable_quad():
    slsys1, slsys2 = quadrotor_as_separable()
    q_functon = partial(fake_proximal_q_function, slsys1)
    planning_from_learned_q_function(q_functon, slsys2)
