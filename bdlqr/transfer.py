from collections import deque
from functools import partial
from itertools import zip_longest

import numpy as np
from numpy.linalg import norm
from bdlqr.separable import SeparableLinearSystem, joint_linear_system, solve_seq
from bdlqr.linalg import ScalarQuadFunc


def mask_env_dynamics(slsys):
    return SeparableLinearSystem()


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
    >>> y0 = rng.rand(yD)
    >>> x0 = rng.rand(xD)
    >>> us = [rng.rand(uD) for _ in range(slsys.T)]
    >>> ys, xs = slsys.forward(y0, x0, us)
    >>> V = independent_cost_to_go(slsys)
    >>> (V(y0) > sum(yt.T.dot(slsys.Qy).dot(yt) for yt in ys)).all()
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
    >>> y0 = rng.rand(yD)
    >>> x0 = rng.rand(xD)
    >>> us = [rng.rand(uD) for _ in range(slsys.T)]
    >>> ys, xs = slsys.forward(y0, x0, us)
    >>> vD = slsys.Bv.shape[-1]
    >>> vks = [rng.rand(vD)]
    >>> ρ = 1
    >>> GV = independent_cost_to_go(slsys)
    >>> V = proximal_cost_to_go(slsys, GV, vks, ρ)
    >>> (V(y0) > sum(yt.T.dot(slsys.Qy).dot(yt) +
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
    vh  = np.hstack((v, 1))
    vhD = vh.shape[0]
    # V(y, ṽ) = arg min_v ∑_ṽ 0.5 ρ |ṽ_t - v_t]_2^2 + ∑_t y Q y
    #            y_t+1   = Ay y_{t}   + Bv v_t
    # V(y, ṽ) = arg min_v ∑ y Q y + 0.5 ρ [v_t^T, 1] [    1, -ṽ_t]  [v_t ]
    #                                                [-ṽ_t^T,  ṽ^2]  [   1]
    #            y_t+1   = Ay y_{t}   + [Bv, 0] [v_t]
    #                                           [  1]
    sy  = np.zeros(Qy.shape[0])
    zv  = np.zeros(vh.shape[0])
    oyT = np.zeros(QyT.shape[0])
    Bvh = np.hstack((Bv, np.zeros((Bv.shape[0], 1))))
    Rsh = [0.5 * ρ * np.vstack((np.hstack((np.eye(vh.shape[0]),    - ṽt)),
                                np.hstack((             - ṽt.T, norm(ṽt)))))
           for ṽt in ṽk]
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
    return (y.T.dot(Qy).dot(y) + 0.5 * ρ * norm(v - ṽs[0]) +
            proximal_cost_to_go(slsys, independent_cost_to_go(slsys), ṽs[1:], ρ))

def planning_from_learned_q_function(Q1, slsys1, slsys2, y0, x0, vs, us):
    def prox_g_u(xk, zk, wk, ρ):
        return Q1(y0, E.dot(xk) + wk/ρ

    return admm((Q1, slsys2.prox_g_u)


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
