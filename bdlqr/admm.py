import doctest
from functools import partial
from logging import basicConfig, getLogger, DEBUG, INFO
basicConfig()
LOG = getLogger(__name__)
LOG.setLevel(INFO)

import numpy as np
from numpy.linalg import norm

from bdlqr.linalg import ScalarQuadFunc, AffineFunction


def admm_wk(xk, zk, const_fn, grads):
    # Estimate wk for dual_feasibility_check
    if grads:
        assert isinstance(const_fn, AffineFunction)
        A = const_fn.A[:, :xk.shape[0]]
        B = const_fn.A[:, xk.shape[0]:]
        wk, err, _,_ = np.linalg.lstsq(
            np.vstack((A.T, B.T)),
            - np.hstack((grads[0](xk),
                         grads[1](zk))),
            rcond=None)
    else:
        assert isinstance(const_fn, AffineFunction)
        wk = np.zeros_like(const_fn.B)

    return wk


def admm(proximals, x0, z0, w0, const_fn, ρ,
         dual_feasibility_check=None, objs=(), grads=(),
         max_iter=100, thresh=1e-4, debug_print_len=3):
    """
    Run admm for

    minimize f(x) + g(z)
        s.t. Ax + Bz = c

    @params
    argmin_Lps = (argmin_Lp_x, argmin_Lp_z)

    # Iterates over
    """
    xk = x0
    zk = z0
    wk = w0
    dpk = min(debug_print_len, len(x0))

    for k in range(max_iter):
        if dual_feasibility_check:
            dual_feasibility_check(xk, zk, wk, err[0])
        xkp1 = proximals[0](xk, zk, wk, ρ)
        LOG.debug(" x%d[:%d]=%s", k, dpk, xkp1[:dpk])
        zkp1 = proximals[1](xkp1, zk, wk, ρ)
        LOG.debug(" z%d[:%d]=%s", k, dpk, zkp1[:dpk])
        wk   = wk + ρ*const_fn(np.hstack((xkp1, zkp1)))
        LOG.debug(" w%d[:%d]=%s", k, dpk, wk[:dpk])
        change = norm(xk - xkp1) + norm(zk - zkp1)
        xk = xkp1
        zk = zkp1
        if LOG.level <= DEBUG:
            if objs:
                LOG.debug(" f(x)=%0.03f, g(z)=%0.03f", objs[0](xk) , objs[1](zk))
            LOG.debug(" |Ax+Bz-c|=%0.03f", norm(const_fn(np.hstack((xk, zk)))[0]))
        # TODO: Make the breaking criterion depend on primal and dual residual instead of
        # norms
        if change < thresh:
            LOG.debug("breaking after {} < {} iterations with change = {} < {}"
                      .format(k, max_iter, change, thresh))
            break
        if (
                any((np.isnan(a).any() for a in [xk, zk, wk]))
                or any((np.isnan(a).any() for a in [xk, zk, wk]))
        ):
            raise RuntimeError(
                "Either of xk, zk, wk is nan or inf:\n xk={}, zk={}, wk={}"
                .format(xk=xk, zk=zk, wk=wk))
    return xk, zk, wk



class QuadraticADMM:
    """
    minimize xᵀQx + 2sᵗx + zᵀRz + 2uᵀz
    s.t. Ax + Bz = c
    """
    def __init__(self, Q, s, R, u, A, B, c):
        self.obj_x = ScalarQuadFunc(Q, s, 0)
        self.obj_z =  ScalarQuadFunc(R, u, 0)
        self.obj   = self.obj_x.add_concat(self.obj_z)
        self.constraint = AffineFunction(np.hstack(([A, B])), -c)

    def augmented_lagrangian(self, ρ):
        wD = self.constraint.B.shape[0]
        w_zeroq = ScalarQuadFunc.zero(wD)
        w = AffineFunction(np.eye(wD), np.zeros(wD))
        penalty = 0.5 * ρ * self.constraint.dot(self.constraint)
        return (self.obj.add_concat(w_zeroq) +
                self.constraint.mul_concat(w) +
                penalty.add_concat(w_zeroq))

    def Lp(self, w, ρ):
        """
        represent f(x) + g(z) + wᵀ(Ax+Bz-c) + 0.5ρ|Ax+Bz-c|₂²
        as quadratic in x, z
        xᵀLQx x + 2*zᵀ LQxz x + zᵀ LQz z + 2 lxᵀ x + 2 lzᵀ z

        Return LQx, LQxz, LQxz, lx, lz and ls
        """
        return self.obj + self.constraint.dot(w) + 0.5 * ρ * self.constraint.dot(self.constraint)

    def eval_Lp_raw(self, x, z, w, ρ):
        xD = self.obj_x.l.shape[0]
        zD = self.obj_z.l.shape[0]
        wD = self.constraint.B.shape[0]
        Q = self.obj_x.Q
        s = self.obj_x.l
        R = self.obj_z.Q
        u = self.obj_z.l
        A = self.constraint.A[:, :xD]
        B = self.constraint.A[:, xD:]
        c = self.constraint.B
        const = A.dot(x)+B.dot(z)-c
        Lp_val_raw = (x.T.dot(Q).dot(x)
                      , 2*s.T.dot(x)
                      , z.T.dot(R).dot(z)
                      , 2*u.T.dot(z)
                      , w.T.dot(const)
                      , 0.5*ρ*(const.T.dot(const)))
        return Lp_val_raw

    def eval_Lp(self, x, z, w, ρ):
        """
        >>> qadmm = QuadraticADMM(*random_bi_quadratic2(xD=1, zD=1, cD=1))
        >>> lp = sum(qadmm.eval_Lp(*map(np.array, (1, 0, 0, 1))))
        >>> lp_exp = sum(qadmm.eval_Lp_raw(*map(np.array, (1, 0, 0, 1))))
        >>> np.allclose(lp , lp_exp)
        True
        >>> lp = sum(qadmm.eval_Lp(*map(np.array, (0, 1, 0, 1))))
        >>> lp_exp = sum(qadmm.eval_Lp_raw(*map(np.array, (0, 1, 0, 1))))
        >>> np.allclose(lp , lp_exp)
        True
        >>> lp = sum(qadmm.eval_Lp(*map(np.array, (0, 0, 1, 1))))
        >>> lp_exp = sum(qadmm.eval_Lp_raw(*map(np.array, (0, 0, 1, 1))))
        >>> np.allclose(lp , lp_exp)
        True
        >>> lp = sum(qadmm.eval_Lp(*map(np.array, (1, 1, 0, 1))))
        >>> lp_exp = sum(qadmm.eval_Lp_raw(*map(np.array, (1, 1, 0, 1))))
        >>> np.allclose(lp , lp_exp)
        True
        >>> lp = sum(qadmm.eval_Lp(*map(np.array, (1, 0, 1, 1))))
        >>> lp_exp = sum(qadmm.eval_Lp_raw(*map(np.array, (1, 0, 1, 1))))
        >>> np.allclose(lp , lp_exp)
        True
        >>> lp = sum(qadmm.eval_Lp(*map(np.array, (0, 1, 1, 1))))
        >>> lp_exp = sum(qadmm.eval_Lp_raw(*map(np.array, (0, 1, 1, 1))))
        >>> np.allclose(lp , lp_exp)
        True
        >>> lp = sum(qadmm.eval_Lp(*map(np.array, (1, 1, 1, 1))))
        >>> lp_exp = sum(qadmm.eval_Lp_raw(*map(np.array, (1, 1, 1, 1))))
        >>> np.allclose(lp , lp_exp)
        True
        >>> x, z, w, ρ = map(np.array, np.random.rand(4))
        >>> lp = sum(qadmm.eval_Lp(x, z, w, ρ))
        >>> lp_exp = sum(qadmm.eval_Lp_raw(x, z, w, ρ))
        >>> np.allclose(lp , lp_exp)
        True
        """
        return self.Lp(w, ρ)(np.hstack((x, z)))
        LQx, LQxz, LQz, lx, lz, cnst = self.Lp(w, ρ)
        Lp_val = (x.dot(LQx).dot(x)
                  , z.dot(LQxz).dot(x)
                  , z.dot(LQz).dot(z)
                  , 2*lx.dot(x)
                  , 2*lz.dot(z)
                  , cnst)
        return Lp_val


    def prox_fx(self, _, z, w, ρ):
        quad_xyw = self.Lp(w, ρ)
        xD = self.obj_x.l.shape[0]
        return quad_xyw.partial(0, xD, np.array([]), z).argmin()

    def prox_gz(self, x, _, w, ρ):
        quad_xyw = self.Lp(w, ρ)
        xD = self.obj_x.l.shape[0]
        zD = self.obj_z.l.shape[0]
        return quad_xyw.partial(xD, xD+zD, x, np.array([])).argmin()

    def solve_admm(self, ρ=0.1, max_iter=100, thresh=1e-4):
        xD = self.obj_x.l.shape[0]
        A  = self.constraint.A[:, :xD]
        B  = self.constraint.A[:, xD:]
        c  = self.constraint.B
        x0 = self.obj_x.argmin()
        z0 = self.obj_z.argmin()
        const_fn = AffineFunction(np.hstack((A,B)), c)
        w0 = admm_wk(x0, z0, const_fn, (self.grad_f_x, self.grad_g_z))
        return admm((self.prox_fx, self.prox_gz),
                    x0, z0, w0, const_fn, ρ,
                    objs=(self.obj_x, self.obj_z),
                    max_iter=max_iter, thresh=thresh)

    def grad_f_x(self, x):
        return self.obj_x.grad()(x)

    def grad_g_z(self, z):
        return self.obj_z.grad()(z)

    def dual_feasibility_check(self, x, z, w, err):
        xD = self.obj_x.l.shape[0]
        assert np.allclose(self.constraint.A[:, :xD].T.dot(w) + self.grad_f_x(x), 0, atol=err)
        assert np.allclose(self.constraint.A[:, xD:].T.dot(w) + self.grad_g_z(z), 0, atol=err)

    def solve(self, test=True, ρ=0.1):
        """
        minimize xᵀQx + 2sᵗx + zᵀRz + 2uᵀz + wᵀ(Ax + Bz - c)

        Represent as a quadratic in xzw = [x, z, w]
        And then solve as lstsq
        """
        xD = self.obj_x.l.shape[0]
        zD = self.obj_z.l.shape[0]
        xzwopt = self.augmented_lagrangian(ρ).argmin()
        return xzwopt[:xD], xzwopt[xD:xD+zD], xzwopt[xD+zD:]


def random_bi_quadratic2(xD = 1,
                     zD = 1,
                     cD = 1,
                     xopt = -1,
                     zopt = 1):
    Qsqrt = np.random.rand(xD,xD)
    Q = Qsqrt.T.dot(Qsqrt)
    s = - Qsqrt.dot(np.ones(xD))
    Rsqrt = np.random.rand(zD, zD)
    R = Rsqrt.T.dot(Rsqrt)
    u = Rsqrt.dot(np.ones(zD))
    A = np.eye(xD) # np.random.rand(cD, xD)
    B = -np.eye(xD) # np.random.rand(cD, zD)
    c = np.zeros(xD) # np.random.rand(cD)
    return Q, s, R, u, A, B, c


def random_bi_quadratic(maxD=100, rng=np.random):
    quad1 = ScalarQuadFunc.randomps(rng=rng, xD=rng.randint(1, maxD))
    quad2 = ScalarQuadFunc.randomps(rng=rng, xD=rng.randint(1, maxD))
    cD = rng.randint(1, maxD)
    A = rng.rand(cD, quad1.domain_size)
    B = rng.rand(cD, quad2.domain_size)
    c = rng.rand(cD)
    return quad1.Q, quad1.l, quad2.Q, quad2.l, A, B, c


def test_quadratic(example=random_bi_quadratic,
                   ρ=1.0,
                   seed=None,
                   thresh = 1e-2):
    if seed is None:
        seed = np.random.randint(1000000)
    LOG.info("seed = {}".format(seed))
    rng = np.random.RandomState(seed)
    ex = list(example(rng=rng))
    LOG.info("Testing {}".format( ex ))
    qadmm = QuadraticADMM(*ex)
    xopt_admm, zopt_admm, wopt_admm = qadmm.solve_admm(
        ρ=ρ,
        max_iter=10000)
    xopt, zopt, wopt = qadmm.solve(ρ=ρ)
    assert np.allclose(xopt, xopt_admm, atol=thresh, rtol=thresh)
    assert np.allclose(zopt, zopt_admm, atol=thresh, rtol=thresh)


def unit_quadratic(rng=None):
    # f(x) = (x + 1)^2
    # g(z) = (z - 1)^2
    Q = [[1]]
    s = [1]
    R = [[1]]
    u = [-1]
    A = [[1]]
    B = [[-1]]
    c = [0]
    return map(np.array, (Q, s, R, u, A, B, c))


def custom_quadratic(rng=None):
    # f(x) = (x + 1)^2
    # g(z) = (z - 1)^2
    Q = [[0.25]]
    s = [0.5]
    R = [[0.25]]
    u = [-0.5]
    A = [[1]]
    B = [[-1]]
    c = [0]
    return map(np.array, (Q, s, R, u, A, B, c))


test_custom_quadratic = partial(test_quadratic,
                                example=custom_quadratic)

if __name__ == '__main__':
    doctest.testmod()
    test_quadratic(example=unit_quadratic)
    test_quadratic(example=custom_quadratic, ρ=1)
    test_quadratic(ρ=1)


