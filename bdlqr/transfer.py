from collections import deque
from functools import partial
from itertools import zip_longest, product
from logging import basicConfig, getLogger, INFO, DEBUG
basicConfig()
LOG = getLogger(__name__)
LOG.setLevel(DEBUG)

import numpy as np
from numpy.linalg import norm
from scipy.optimize import fmin
from kwplus.functools import recpartial

from bdlqr.separable import (SeparableLinearSystem, joint_linear_system,
                             plot_separable_sys_results, list_extendable,
                             solve_full, solve_seq, plotables)
from bdlqr.linalg import ScalarQuadFunc
from bdlqr.lqr import LinearSystem
from bdlqr.admm import admm
from bdlqr.functoolsplus import getdefaultkw


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
    return LinearSystem(Ay, Bv, Qy, sy, Rv, zv, QyT, sy, T, γ=slsys.γ)


def independent_cost_to_go(slsys, max_iter=100):
    """
    V(y₀) = min_v ∑ₜ c(yₜ)
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
    ufs = list(ufs)
    Vfs = list(Vfs)
    return Vfs


def no_upper_bound_factor(slsys, t):
    return 1


def tri_ineq_upper_bound_factor(slsys, t):
    """
    Assume ||xₜ₊₁ - xₜ|| ≤ ϵ and ||v*ₜ₊₁ - v*ₜ|| ≤ δ

    Then ||ṽₜ₊ₖ - v*ₜ₊ₖ|| ≤ kEϵ + kδ + ||ṽₜ - v*ₜ||
    And ∑ₖ₌ₜᵀ ||ṽₖ - v*ₖ|| ≤ (T-t+1)(kEϵ + kδ + ||ṽₜ - v*ₜ||)
    """
    return (slsys.T-t+1)


def lypunov_exponent_upper_bound_factor(slsys, t, λ=-0.5):
    """
    Assume |ṽₜ - vₜ| ≈ e^λᵗ |ṽ₀ - v₀|

    And ∑ₖ₌ₜᵀ ||ṽₖ - v*ₖ|| ≤ (∑ₖ₌₀ᵀ⁻ᵗ exp(λt)) ||ṽ₀-v*₀||

    (∑ₖ₌₀ᵀ⁻ᵗ exp(λt)) = (exp(λ(T-t)) - 1) / (exp(λ) - 1)
    """
    T = slsys.T
    return (1 - np.exp(λ*(T-t+1))) / (1-np.exp(λ))


def proximal_env_linsys(slsys, ṽks, ρ, t,
                        upper_bound_factor=no_upper_bound_factor):
    """

    Vₚᵣₒₓ(y₁, ṽs₁) = minᵥ ∑ⁿₜ₌₁ yₜᵀQyₜ + 0.5 ρ T |vₜ - ṽₜ| + GV(yₖ₊₁)
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
    # V(y, ṽ) = arg min_v ∑_ṽ 0.5 ρ T |ṽₜ - vₜ]_2^2 + ∑ₜ y Q y
    #            yₜ₊₁   = Ay yₜ   + Bv vₜ
    # V(y, ṽ) = arg min_v ∑ y Q y + 0.5 ρ T [vₜᵀ, 1] [    1, -ṽₜ]  [vₜ ]
    #                                                [-ṽₜᵀ,  ṽ^2]  [   1]
    #            yₜ₊₁   = Ay yₜ   + [Bv, 0] [vₜ]
    #                                       [ 1]
    sy  = np.zeros(yD)
    oyT = np.zeros(yD)
    ρT = ρ * upper_bound_factor(slsys, t)
    Rsv = [0.5 * ρT * np.eye(vD) for _ in ṽks]
    zsv = [-0.5 * ρT * ṽt for ṽt in ṽks]

    # IV(y₀) := arg min_v ∑ₜ yₜᵀQyₜ
    #         s.t.  yₜ₊₁ = Ay yₜ + Bv vₜ   ∀ t
    if slsys.T > t + len(ṽks):
        slsys_remaining = SeparableLinearSystem.copy(slsys,
                                                     T=slsys.T - t - len(ṽks))
        IV, *_ = independent_cost_to_go(slsys_remaining)
    else:
        IV = ScalarQuadFunc(np.zeros_like(Qy), np.zeros_like(sy), np.zeros(1))
    return LinearSystem(Ay, Bv, Qy, sy, Rsv, zsv, IV.Q, IV.l, len(Rsv), γ=slsys.γ)


def proximal_robot_linsys(mslsys, vks, wks, ρ, t,
                          upper_bound_factor=no_upper_bound_factor):
    """

    Vₚᵣₒₓ(xₜ, vsₜ₊₁:) = min_u uᵀRu + 0.5 ρ |Exₜ₊₁(u) + wₜ₊₁/ρ - vₜ₊₁|₂² + ∑ⁿₜ₌₁ uₜᵀRuₜ
    """
    # vks = [vₜ₊₁]
    # wks = [wₜ₊₁]
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
    # Qx = 0.5 ρ [           EᵀE,                    -Eᵀvₜ₊₁ + Eᵀwₜ₊₁/ρ]
    #            [-vₜ₊₁ᵀE + wₜ₊₁ᵀE/ρ, (-vₜ₊₁ + wₜ₊₁/ρ)ᵀ(-vₜ₊₁ + wₜ₊₁/ρ)]
    xhD = xD + 1
    vdesired = [(-vtp1 + wtp1/ρ)[:, None]
                for vtp1, wtp1 in zip(vks, wks)]
    ρT = ρ * upper_bound_factor(mslsys, t)
    Qxhs = [0.5 * ρT * np.vstack((np.hstack((E.T.dot(E),     E.T.dot(vdtp1))),
                                  np.hstack((vdtp1.T.dot(E), vdtp1.T.dot(vdtp1)))))
            for vdtp1 in vdesired]
    sxh  = np.zeros(xhD)
    zu  = np.zeros(uD)
    oyT = np.zeros(xhD)
    Axh = np.eye(xhD)
    Axh[:-1, :-1] = Ax
    Buh = np.vstack((Bu, np.zeros((1, uD))))
    return LinearSystem(Axh, Buh, Qxhs[:-1], sxh, R, zu, Qxhs[-1], sxh, len(Qxhs), γ=mslsys.γ)


def proximal_robot_solution(mslsys, vks, wks, ρ, t,
                            proximal_robot_linsys_=proximal_robot_linsys):
    sys_h = proximal_robot_linsys_(mslsys, vks, wks, ρ, t)
    ufhs, Vfhs = sys_h.solve_f(return_min=True)
    return list(ufhs), list(Vfhs)


def proximal_env_solution(slsys, ṽks, ρ, t,
                          proximal_env_linsys_=proximal_env_linsys):
    """

    Vₚᵣₒₓ(y₁, ṽs₁) = minᵥ ∑ⁿₜ₌₁ yₜᵀQyₜ + 0.5 ρ T |vₜ - ṽₜ| + GV(yₖ₊₁)

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
    prox_env_sys = proximal_env_linsys_(slsys, ṽks, ρ, t)
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
                           T   = np.Inf,
                           γ   = 1):
    Bu1=[[1/m1]]
    R1=[[r01]]
    Bu2=[[1/m2]]
    R2=[[r02]]
    QyT = np.array(Qy)*100
    return (SeparableLinearSystem(*map(np.array, (Qy, R1, Ay, Bv, QyT, E1, Ax1, Bu1, T, γ))),
            SeparableLinearSystem(*map(np.array, (Qy, R2, Ay, Bv, QyT, E2, Ax2, Bu2, T, γ))))


def solve_mpc_admm(argmin_Q1, mslsys2, yt, xt, ρ, t, admm_=admm, k_mpc=1,
                   proximal_robot_solution_=proximal_robot_solution):
    """
    """
    if k_mpc != 1: raise NotImplementedError()
    uD = mslsys2.Bu.shape[-1]
    vD = mslsys2.E.shape[0]
    E  = mslsys2.E

    # vt is determined
    vt = E.dot(xt)
    # yₜ₊₁ is determined but cannot be used, because we do not know environment
    # dynamics:
    # yₜ₊₁ = f(yₜ, vₜ)
    # We are solving for uₜ, vₜ₊₁, xₜ₊₁, yₜ₊₂
    def prox_v(_, us, ws, ρ):
        # _  = vs = [vₜ₊₁]
        # us = [uₜ]
        # ws = [wₜ₊₁]
        if len(us) != k_mpc: raise NotImplementedError()
        if len(ws) != k_mpc: raise NotImplementedError()
        ṽs = [E.dot(mslsys2.f_x(xt, us[0])) + ws[0]/ρ]

        # We want argmin_Q1 to give two step ahead prediction
        # vₜ₊₁ = arg minᵥ vₜ R vₜ + V(yₜ₊₂(yₜ, uₜ))
        vtp1 = argmin_Q1(yt, vt, ṽs, t)
        vs = vtp1.reshape(1, -1)
        # vs = [vₜ₊₁]
        return vs

    def prox_u(vs, _, ws, ρ):
        # vs = [vₜ₊₁]
        # _ = us = [uₜ]
        # ws = [wₜ₊₁]
        if len(vs) != k_mpc: raise NotImplementedError()
        if len(ws) != k_mpc: raise NotImplementedError()
        ufhs, Vfhs = proximal_robot_solution_(mslsys2, vs, ws, ρ, t)
        ufh0 = ufhs[0]
        us = ufh0(np.hstack((xt, [1]))).reshape(1, -1)
        # us = [uₜ]
        return us

    us0 = np.zeros((1, uD))
    ws0 = np.zeros((1, vD))
    vs0 = E.dot(xt).reshape(1, -1)
    const_ = partial(mslsys2.constraint_fn, yt, xt)
    vs, us, ws = admm_((prox_v, prox_u), vs0, us0, ws0, const_, ρ)
    return us[0]


def training_system_flexible(slsys2):
    return SeparableLinearSystem.copy(
            slsys2,
            R=np.zeros_like(slsys2.R),
            Ax=np.zeros_like(slsys2.Ax),
            Bu=np.ones_like(slsys2.Bu))


def training_system_equal(slsys2):
    return SeparableLinearSystem.copy(slsys2)


def transfer_mpc_admm(slsys1, slsys2, y0, x0, traj_len, ρ=1,
                      solve_mpc_admm_=solve_mpc_admm,
                      proximal_env_solution_=proximal_env_solution):
    """
    @param solve_mpc_admm_: Pre-configured solve_mpc_admm
    """
    if slsys1 is None:
        slsys1 = training_system_equal(slsys2)

    def argmin_Q1(yt, vt, ṽks, t):
        # ṽks = [ṽₜ₊₁]
        # vₜ₊₁ = arg minᵥ vₜ R vₜ + V(yₜ₊₂(yₜ, uₜ))
        ytp1 = slsys1.f_y(yt, vt)
        vfs, Vfs = proximal_env_solution_(slsys1, ṽks, ρ, t+1)
        return vfs[0](ytp1)

    mslsys1 = MaskEnvDynamics(slsys1)
    mslsys2 = MaskEnvDynamics(slsys2)
    us = []
    ys = [y0]
    xs = [x0]
    for t in range(traj_len):
        ut = solve_mpc_admm_(argmin_Q1, mslsys2, ys[t], xs[t], ρ, t)
        us.append(ut)
        xs.append(slsys2.f_x(xs[t], ut))
        vt = slsys2.effect(xs[t])
        ys.append(slsys2.f_y(ys[t], vt))

    ys, xs = slsys2.forward(y0, x0, us)
    return ys, xs, us


def transfer_mpc_admm_set_upper_bound(transfer_mpc_admm_=transfer_mpc_admm,
                                      upper_bound_factor=no_upper_bound_factor,
                                      name="transfer_mpc_admm"):
    partialfunc = recpartial(
        transfer_mpc_admm_,
        {"solve_mpc_admm_.proximal_robot_solution_.proximal_robot_linsys_.upper_bound_factor": upper_bound_factor,
         "proximal_env_solution_.proximal_env_linsys_.upper_bound_factor": upper_bound_factor})
    partialfunc.__name__ = name
    return partialfunc


tri_ineq_transfer_mpc_admm = transfer_mpc_admm_set_upper_bound(
    upper_bound_factor=tri_ineq_upper_bound_factor,
    name="tri_ineq_transfer_mpc_admm")


lypunov_transfer_mpc_admm = transfer_mpc_admm_set_upper_bound(
    upper_bound_factor=lypunov_exponent_upper_bound_factor,
    name="lypunov_transfer_mpc_admm")


def example_multi_dim_double_integrator(r0 = 1,
                                        Ay = np.eye(2),
                                        Bv = np.eye(2),
                                        E  = [[1., 1, 0, 0],
                                              [0., 0, 1, 1]],
                                        Qy = np.eye(2),
                                        Ax = [[0., 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1],
                                              [0, 0, 0, 0]],
                                        Bu = [[0.],
                                              [0],
                                              [0],
                                              [1]],
                                        y0 = [-1],
                                        x0 = [0.],
                                        T  = 30,
                                        γ  = 1):
    R=[[r0]]
    QyT = np.asarray(Qy)*100
    xD = np.asarray(Ax).shape[-1]
    yD = np.asarray(Ay).shape[-1]
    x0 = np.ones(xD) * x0
    y0 = np.ones(yD) * y0
    return [plotables] + list(map(np.asarray, (y0, x0, Qy, R, Ay, Bv, QyT, E, Ax, Bu, T, γ)))

def plot_separable_sys_results_configs(parent_conf={}, configs_gen=list):
    parent_conf.update({
        "example.T": 40,
        "getsolvers_":
        list_extendable(
            [solve_full,
             solve_seq,
             partial(
                 transfer_mpc_admm,
                 None),
             partial(
                 tri_ineq_transfer_mpc_admm,
                 None),
             partial(
                 lypunov_transfer_mpc_admm,
                 None)
            ])
    })

    configs = configs_gen()
    pow10 = partial(np.float_power, 10)
    for r0 in map(pow10, range(-2, 5)):
        ccopy = parent_conf.copy()
        ccopy.update({
                "example.r0": r0
        })
        configs.append(ccopy)


    for x0, y0 in product([0.1, 0.0, -0.1], repeat=2):
        ccopy = parent_conf.copy()
        ccopy.update({
                "example.y0": [y0],
                "example.x0": [x0]
        })
        configs.append(ccopy)

    return configs


plot_separable_sys_results_multi_dim_configs = partial(
    plot_separable_sys_results_configs,
    parent_conf={"example":example_multi_dim_double_integrator})


def main(func=plot_separable_sys_results,
         configs_gen=plot_separable_sys_results_multi_dim_configs):
    for conf in configs_gen():
        recpartial(func, conf)()

if __name__ == '__main__':
    main()
