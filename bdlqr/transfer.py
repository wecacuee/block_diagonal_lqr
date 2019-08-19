from collections import deque
from functools import partial
from itertools import zip_longest
from logging import basicConfig, getLogger, INFO
basicConfig()
LOG = getLogger(__name__)
LOG.setLevel(INFO)

import numpy as np
from numpy.linalg import norm
from scipy.optimize import fmin

from bdlqr.separable import (SeparableLinearSystem, joint_linear_system,
                             plot_separable_sys_results)
from bdlqr.linalg import ScalarQuadFunc
from bdlqr.lqr import affine_backpropagation, LinearSystem
from bdlqr.admm import admm
from bdlqr.functools import getdefaultkw



class MaskEnvDynamics:
    def __init__(self, slsys):
        self._slsys = slsys

    def Ay(self):
        raise NotImplementedError("Env dynamics not allowed")

    def Bv(self):
        raise NotImplementedError("Env dynamics not allowed")

    def __getattr__(self, a):
        return getattr(self._slsys, a)


def independent_env_linsys(slsys):
    """
    V(y₀) = arg min_v ∑ₜ c(yₜ)
            s.t.  yₜ₊₁ = Ay yₜ + Bv vₜ   ∀ t
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
    return LinearSystem(Ay, Bv, Qy, sy, Rv, zv, QyT, sy, T)


def independent_cost_to_go(slsys, max_iter=100):
    """
    V(y₀) = arg min_v ∑ₜ c(yₜ)
            s.t.  yₜ₊₁ = Ay yₜ + Bv vₜ   ∀ t

    Let Vₜ(yₜ) = yₜᵀ Pₜ yₜ + 2 oₜᵀyₜ
    return ScalarQuadFunc(Pₜ, oₜ, 0)

    >>> seed = np.random.randint(10000)
    >>> LOG.info("seed={}".format(seed))
    >>> rng = np.random.RandomState(seed)
    >>> slsys = SeparableLinearSystem.random(T=3, rng=rng)
    >>> yD = slsys.Ay.shape[-1]
    >>> vD = slsys.Bv.shape[-1]
    >>> y0 = rng.rand(yD)
    >>> vs = [rng.rand(vD) for _ in range(slsys.T)]
    >>> ys = slsys.forward_y(y0, vs)
    >>> Vf = independent_cost_to_go(slsys)[0]
    >>> (Vf(y0) <= ys[-1].T.dot(slsys.QyT).dot(ys[-1]) + sum(
    ...             yt.T.dot(slsys.Qy).dot(yt) for yt in [y0] + ys[:-1])).all()
    True
    """
    ind_env_linsys = independent_env_linsys(slsys)
    ufs, Vfs = ind_env_linsys.solve_f(max_iter=max_iter, return_min=True)
    return list(Vfs)


def proximal_env_linsys(slsys, ṽks, ρ, t):
    """

    Vₚᵣₒₓ(y₁, ṽs₁) = minᵥ ∑ⁿₜ₌₁ yₜᵀQyₜ + 0.5 ρ |vₜ - ṽₜ| + GV(yₖ₊₁)
    """
    n = len(ṽks)
    Qy   = slsys.Qy
    Ay   = slsys.Ay
    Bv   = slsys.Bv
    T    = slsys.T
    assert len(ṽks) <= T

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
    oyT = np.zeros(yD)
    Rsv = [0.5 * ρ * np.eye(vD) for _ in ṽks]
    zsv = [-0.5 * ρ * ṽt for ṽt in ṽks]

    # IV(y₀) := arg min_v ∑ₜ yₜᵀQyₜ
    #         s.t.  yₜ₊₁ = Ay yₜ + Bv vₜ   ∀ t
    slsys_remaining = SeparableLinearSystem.copy(slsys,
                                                 T=slsys.T - t - len(ṽks))
    IV, *_ = independent_cost_to_go(slsys_remaining)
    return LinearSystem(Ay, Bv, Qy, sy, Rsv, zsv, IV.Q, IV.l, len(Rsv))


def proximal_robot_linsys(mslsys, vks, wks, ρ):
    """

    Vₚᵣₒₓ(xₜ, vsₜ₊₁:) = min_u uᵀRu + 0.5 ρ |Exₜ₊₁(u) + wₜ₊₁/ρ - vₜ₊₁|₂² + ∑ⁿₜ₌₁ uₜᵀRuₜ
    """
    n = len(vks)
    E    = mslsys.E
    R    = mslsys.R
    Ax   = mslsys.Ax
    Bu   = mslsys.Bu
    T    = mslsys.T
    assert len(vks) <= T

    xD  = Ax.shape[-1]
    uD  = Bu.shape[-1]
    # Vₚᵣₒₓ(xₜ, vsₜ₊₁:) = min_u uᵀRu + 0.5 ρ |Exₜ₊₁(u) + wₜ₊₁/ρ - vₜ₊₁|₂² + ∑ⁿₜ₌₁ uₜᵀRuₜ
    # Qx = 0.5 ρ [           EᵀE,                    -vₜ₊₁ + wₜ₊₁/ρ]
    #            [-vₜ₊₁ + wₜ₊₁/ρ, (-vₜ₊₁ + wₜ₊₁/ρ)ᵀ(-vₜ₊₁ + wₜ₊₁/ρ)]
    xhD = xD + 1
    vdesired = [(-vtp1 + wtp1/ρ)[:, None]
                for vtp1, wtp1 in zip(vks, wks)]
    Qxhs = [0.5 * ρ * np.vstack((np.hstack((E.T.dot(E), vdtp1)),
                                 np.hstack((vdtp1.T, vdtp1.T.dot(vdtp1)))))
           for vdtp1 in vdesired]
    sxh  = np.zeros(xhD)
    zu  = np.zeros(uD)
    oyT = np.zeros(xhD)
    Axh = np.eye(xhD)
    Axh[:-1, :-1] = Ax
    Buh = np.vstack((Bu, np.zeros((1, uD))))
    return LinearSystem(Axh, Buh, Qxhs[:-1], sxh, R, zu, Qxhs[-1], sxh, len(Qxhs))


def proximal_robot_solution(mslsys, vks, wks, ρ):
    sys_h = proximal_robot_linsys(mslsys, vks, wks, ρ)
    ufhs, Vfhs = sys_h.solve_f(return_min=True)
    ufs = [ufht[:, :-1] for ufht in ufhs]
    Vfs = [Vfht[:-1] for Vfht in ufhs]
    return list(ufs), list(Vfs)


def proximal_env_solution(slsys, ṽks, ρ, t):
    """

    Vₚᵣₒₓ(y₁, ṽs₁) = minᵥ ∑ⁿₜ₌₁ yₜᵀQyₜ + 0.5 ρ |vₜ - ṽₜ| + GV(yₖ₊₁)

    Let Vₜ(yₜ) = yₜᵀ Pₜ(ṽs₁) yₜ + 2 oₜᵀyₜ
    return ScalarQuadFunc(Pₜ, oₜ, 0)

    >>> seed = np.random.randint(10000)
    >>> LOG.info("seed={}".format(seed))
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
    >>> ufs, Vfs = proximal_env_solution(slsys, vks, ρ, 0)
    >>> (Vfs[0](y0) <= ys[-1].dot(slsys.QyT).dot(ys[-1])
    ...               + y0.T.dot(slsys.Qy).dot(y0)
    ...               + sum(yt.T.dot(slsys.Qy).dot(yt) for yt in ys[:-1])
    ...               + sum(0.5 * ρ * norm(vkt - slsys.E.dot(xt))
    ...                     for xt, vkt in zip(xs, vks))).all()
    True
    """
    prox_env_sys = proximal_env_linsys(slsys, ṽks, ρ, t)
    ufs, Vfs = prox_env_sys.solve_f(return_min=True)
    return list(ufs), list(Vfs)


def vtilde_dynamics(slsys, yt, xt, wt):
    """
    return argmin_u uₜᵀRₜuₜ + 0.5*ρ*|Exₜ - vₜ + wₜ/ρ|
                 s.t. xₜ₊₁ = Ax xₜ + Bu uₜ
    """
    E = slsys.E
    ṽt = E.dot(xt)
    u0 = slsys.prox_g_u(None, xt, [ṽt], None, [wt])
    Ax = slsys.Ax
    Bu = slsys.Bu
    return E.dot(Ax.dot(x0) + Bu.dot(u0)) + ws[0]/ρ


def ground_truth_env_dynamics(slsys, yt, vt):
    Ay = slsys.Ay
    Bv = slsys.Bv
    return Ay.dot(yt) + Bv.dot(vt)


def approx_env_dynamics(masked_slsys, Vprox, yt, xt, wt, vt, ρ):
    """
    yₜ₊₁ = arg min_y | Vprox(yₜ, ṽₜ) + yₜᵀQyₜ + 0.5 * ρ |ṽₜ - vₜ|₂² -  Vprox(y) |
    return yₜ₊₁
    """
    Q = masked_slsys.Q
    E = masked_slsys.E
    ṽt = E.dot(xt) + wt/ρ
    cost_t = yt.dot(Q).dot(yt) + 0.5 * ρ * (ṽt-vt).T.dot(ṽt-vt)
    target_value = Vprox(yt, ṽt) + cost_t
    cost = lambda y: target_value - Vprox(y, vtilde_dynamics(slsys, y, vt, [wt], ρ))
    return fmin(cost, yt)


def planning_from_learned_q_function(masked_slsys1, Q1, argmin_Q1,
                                     masked_slsys2,
                                     y0, x0,
                                     env_dynamics=approx_env_dynamics,
                                     vtilde_dynamics=vtilde_dynamics):
    slsys2 = masked_slsys2
    T = slsys2.T
    def prox_env(_, uks, wks, ρ):
        """
        return arg min_{vs} h(xks + wks / ρ)
        """
        assert len(uks) == 1
        assert len(wks) == 1
        Ax = slsys2.Ax
        Bu = slsys2.Bu
        E = slsys2.E
        xks, uks = slsys2.forward_x(x0, uks)
        # wt is defined as
        # wₜᵀ(Exₜ-vₜ)
        ṽks = [E.dot(xt) + wt/ρ
               for xt, wt in zip(xks, wks)]
        yt = y0
        vs = []
        vopt = Q1_argmin(yt, ṽks, ρ)
        return vopt

    prox_robot = partial(slsys2.prox_g_u, y0, x0)
    const_fn = partial(slsys2.constraint_fn, y0, x0)
    E = slsys2.E
    uks = [np.array([0])]
    vks = [E.dot(xt) for xt in xks]
    wks = admm_wk(vks, uks, const_fn)
    vks_admm, uks_admm, wks_admm = admm((prox_env, prox_robot), vks, uks, wks,
                                        const_fn, ρ)
    assert len(uks) == 1
    assert len(wks) == 1
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


def solve_mpc_admm(argmin_Q1, mslsys2, yt, xt, ρ, t):
    """
    """
    uD = mslsys2.Bu.shape[-1]
    vD = mslsys2.E.shape[0]
    E = mslsys2.E
    def prox_v(_, us, ws, ρ):
        ṽs = [E.dot(mslsys2.f_x(xt, us[0])) + ws[0]/ρ]
        vt = argmin_Q1(yt, ṽs, t)
        vs = vt.reshape(1, -1)
        return vs

    def prox_u(vs, _, ws, ρ):
        ufs, Vfs = proximal_robot_solution(mslsys2, vs, ws, ρ)
        uf0 = ufs[0]
        us = uf0(xt).reshape(1, -1)
        return us

    us0 = np.zeros((1, uD))
    ws0 = np.zeros((1, vD))
    vs0 = E.dot(xt).reshape(1, -1)
    const_fn = partial(mslsys2.constraint_fn, yt, xt)
    vs, us, ws = admm((prox_v, prox_u), vs0, us0, ws0, const_fn, ρ)
    return us[0]


def transfer_mpc_admm(slsys1, slsys2, y0, x0, traj_len, ρ=1):
    if slsys1 is None:
        slsys1 = SeparableLinearSystem.copy(
            slsys2,
            R=np.zeros_like(slsys2.R),
            Ax=np.zeros_like(slsys2.Ax),
            Bu=np.ones_like(slsys2.Bu))

    def argmin_Q1(yt, ṽks, t):
        vfs, Vfs = proximal_env_solution(slsys1, ṽks, ρ, t)
        return vfs[0](yt)

    mslsys1 = MaskEnvDynamics(slsys1)
    mslsys2 = MaskEnvDynamics(slsys2)
    us = []
    ys = [y0]
    xs = [x0]
    for t in range(traj_len):
        ut = solve_mpc_admm(argmin_Q1, mslsys2, ys[t], xs[t], ρ, t)
        us.append(ut)
        xs.append(slsys2.f_x(xs[t], ut))
        vt = slsys2.effect(xs[t])
        ys.append(slsys2.f_y(ys[t], vt))

    ys, xs = slsys2.forward(y0, x0, us)
    return ys, xs, us

def solve_by_transfer(slsys1):
    return partial(transfer_mpc_admm, slsys1)

def test_transfer_separable_quad():
    slsys1, slsys2 = quadrotor_as_separable()
    y0 = np.array([0.])
    x0 = np.array([0.])
    ρ = 1
    traj_len = 3
    transfer_mpc_admm(slsys1, slsys2, y0, x0, traj_len)

if __name__ == '__main__':
    plot_separable_sys_results(
        solvers=list(getdefaultkw(plot_separable_sys_results, "solvers")) +
        [solve_by_transfer(None)])
