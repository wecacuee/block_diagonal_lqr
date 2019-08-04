import doctest
from functools import partial
from logging import basicConfig, getLogger, DEBUG
basicConfig()
LOG = getLogger(__name__)
LOG.setLevel(DEBUG)

import numpy as np


def admm(argmin_Lps, dual_feasibility_check, objs, grads,
         x0, z0, A, B, c, ρ, max_iter=100, thresh=1e-4):
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
    wk = ρ*(A.dot(xk)+B.dot(zk)-c)
    for k in range(max_iter):
        #dual_feasibility_check(xk, zk, wk, err[0])
        xkp1 = argmin_Lps[0](xk, zk, wk, ρ)
        zkp1 = argmin_Lps[1](xkp1, zk, wk, ρ)
        wk = wk + ρ*(A.dot(xkp1) + B.dot(zkp1) - c)
        if (np.linalg.norm(xk - xkp1)
            + np.linalg.norm(zk - zkp1)) < thresh:
            break
        xk = xkp1
        zk = zkp1
        LOG.debug(" func vals %0.03f, %0.03f", objs[0](xk) , objs[1](zk))
        LOG.debug(" err %0.03f", np.linalg.norm(A.dot(xk)+B.dot(zk) - c))
    return xk, zk



class QuadraticFunction:
    """
    f: x ↦ R
    """
    def __init__(self, Q, l, c):
        """
        """
        xD = Q.shape[0]
        assert Q.shape == (xD, xD)
        assert l.shape == (xD, )
        self.Q = Q
        self.l = l
        self.c = c

    def __call__(self, x):
        """
        return f(x)
        """
        Q = self.Q
        l = self.l
        c = self.c
        return x.T.dot(Q).dot(x) + 2*l.T.dot(x) + c

    def grad(self):
        """
        return f'
        """
        return AffineFunction(2*self.Q, 2*self.l)

    def add(self, other):
        """
        r(x) = f(x) + g(x)

        return r
        """
        Q = self.Q
        l = self.l
        c = self.c
        if isinstance(other, QuadraticFunction):
            assert self.Q.shape == other.Q.shape
            assert self.l.shape == other.l.shape
            Qo = other.Q
            lo = other.l
            co = other.c
        elif isinstance(other, AffineFunction):
            Qo = 0
            lo = 0.5 * other.A.reshape(-1)
            co = other.b
        else:
            raise NotImplementedError("Unknown type for {}".format(other))
        return QuadraticFunction(Q+Qo, l+lo, c+co)

    __add__ = add

    def __getitem__(self, i):
        if not isinstance(i, slice):
            i = slice(i, i+1, 1)
        return QuadraticFunction(self.Q[i, i], self.l[i], self.c)

    def add_concat(self, other):
        """
        r(x, z) = f(x) + g(z)
        return r
        """
        Q = self.Q
        l = self.l
        c = self.c
        Qo = other.Q
        lo = other.l
        co = other.c

        xD = l.shape[0]
        oD = lo.shape[0]
        rD = xD + oD
        Qr = np.eye(rD)
        Qr[:xD, :xD] = Q
        Qr[xD:, xD:] = Qo
        lr = np.hstack((l, lo))
        cr = c + co
        return QuadraticFunction(Qr, lr, cr)

    __or__ = add_concat

    def concat_concat(self, other):
        """
        r(x, z) = [f(x),
                   g(z)]

        return r
        """
        raise NotImplementedError()

    __and__ = concat_concat

    def argmin(self):
        """
         argmin_x f(x)
        """
        return np.linalg.lstsq(self.Q, -self.l, rcond=None)[0]


    def partial(self, vstart, vend, before, after):
        """
        r_z0(x) = f(x,z0)

        return r_z0
        """
        Q = self.Q
        l = self.l
        c = self.c
        s, e = vstart, vend
        keep = slice(s, e)
        bef = slice(0, s)
        aft = slice(e, None)

        Qr = Q[keep, keep]
        lr = 2*Q[keep, bef].dot(before) + 2*Q[keep, aft].dot(after)
        cr = (before.T.dot(Q[bef, bef]).dot(before)
              + after.T.dot(Q[aft, aft]).dot(after)
              + before.T.dot(Q[bef, aft]).dot(after)
              + after.T.dot(Q[aft, bef]).dot(before)
              + 2*l[bef].T.dot(before)
              + 2*l[aft].T.dot(after))
        return QuadraticFunction(Qr, lr, cr)

    @classmethod
    def zero(cls, D):
        return cls(np.zeros((D,D)), np.zeros(D), 0)

    def __lmul__(self, other):
        Q = self.Q
        l = self.l
        c = self.c
        if isinstance(other, float):
            return QuadraticFunction(other * Q, other * l, other * c)
        else:
            raise NotImplementedError(other)

    def __rmul__(self, other):
        if isinstance(other, float):
            return self.__lmul__(other)
        raise NotImplementedError(other)


class AffineFunction:
    def __init__(self, A, b):
        assert A.shape[0] == b.shape[0]
        self.A = A
        self.b = b

    def __call__(self, x):
        A = self.A
        b = self.b
        return A.dot(x) + b

    def grad(self):
        return A

    def add_concat(self, other):
        """
        r(x, z) = f(x) + g(z)
        """
        return AffineFunction(np.hstack((self.A, other.A)), self.b + other.b)

    def add(self, other):
        """
        r(x) = f(x) + g(x)
        """
        return AffineFunction(self.A + other.A, self.b + other.b)

    __add__ = add

    def __getitem__(self, i):
        if not isinstance(i, slice):
            i = slice(i, i+1, 1)
        return AffineFunction(self.A[i, i], self.b[i])

    def concat_concat(self, other):
        """
        r(x, z) = [f(x),
                   g(z)]
        """
        A = self.A
        b = self.b
        Ao = other.A
        bo = other.b

        xD = b.shape[0]
        oD = bo.shape[0]
        rD = xD + oD

        Ar = np.eye(rD)
        Ar[:xD, :xD] = A
        Ar[xD:, xD:] = Ao

        br = np.hstack((b, bo))
        return AffineFunction(Ar, br)

    __or__ = add_concat

    __and__ = concat_concat

    def dot(self, other):
        """
        r(x) = f(x).T *g(x)
        """
        A = self.A
        b = self.b
        if isinstance(other, np.ndarray):
            if other.ndim < 2:
                other = other.reshape(1, -1)
            return AffineFunction(other.dot(A), other.dot(b))
        Ao = other.A
        bo = other.b
        return QuadraticFunction(A.T.dot(Ao),
                                 b.T.dot(Ao) + bo.T.dot(A),
                                 b.T.dot(bo))
    def __lmul__(self, other):
        A = self.A
        b = self.b
        if isinstance(other, float):
            return AffineFunction(other * A, other * b)

    def __rmul__(self, other):
        if isinstance(other, float):
            return self.__lmul__(other)
        raise NotImplementedError()

    def mul_concat(self, other):
        """
        r(x,z) = f(x).T * g(z)

        >>> f = AffineFunction(np.array([[1, -1]]), np.array([0]))
        >>> g = AffineFunction(np.array([[1.]]), np.array([0]))
        >>> r = f.mul_concat(g)
        >>> isinstance(r, QuadraticFunction)
        True
        >>> r.Q
        array([[ 0. ,  0. ,  0.5],
               [ 0. ,  0. , -0.5],
               [ 0.5, -0.5,  0. ]])
        >>> r.l
        array([0., 0., 0.])
        >>> r.c
        0
        """
        A = self.A
        b = self.b
        Ao = other.A
        bo = other.b

        xD = A.shape[1]
        oD = Ao.shape[1]
        rD = xD + oD
        Qr = np.zeros((rD, rD))
        Qr[:xD, xD:] = 0.5 * A.T.dot(Ao)
        Qr[xD:, :xD] = 0.5 * Ao.T.dot(A)
        lr = 0.5 * np.hstack((bo.T.dot(A), b.T.dot(Ao)))
        cr = b.T.dot(bo)
        return QuadraticFunction(Qr, lr, cr)

    def partial(self, vstart, vend, before, after):
        keep = slice(vstart, vend)
        bef = slice(0, vstart)
        aft = slice(vend, None)
        A = self.A
        b = self.b
        Ar = A[:, keep]
        br = b + A[:, bef].dot(before) + A[:, aft].dot(after)
        return AffineFunction(Ar, br)


class QuadraticADMM:
    """
    minimize xᵀQx + 2sᵗx + zᵀRz + 2uᵀz
    s.t. Ax + Bz = c
    """
    def __init__(self, Q, s, R, u, A, B, c):
        self.obj_x = QuadraticFunction(Q, s, 0)
        self.obj_z =  QuadraticFunction(R, u, 0)
        self.obj   = self.obj_x | self.obj_z
        self.constraint = AffineFunction(A, -c) |  AffineFunction(B, np.zeros_like(c))

    def augmented_lagrangian(self, ρ):
        wD = self.constraint.b.shape[0]
        w_zeroq = QuadraticFunction.zero(wD)
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
        wD = self.constraint.b.shape[0]
        Q = self.obj_x.Q
        s = self.obj_x.l
        R = self.obj_z.Q
        u = self.obj_z.l
        A = self.constraint.A[:, :xD]
        B = self.constraint.A[:, xD:]
        c = self.constraint.b
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
        >>> qadmm = QuadraticADMM(*random_quadratic(xD=1, zD=1, cD=1))
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


    def argmin_x(self, x, z, w, ρ):
        quad_xyw = self.Lp(w, ρ)
        xD = x.shape[0]
        Lp_val = sum(self.eval_Lp(x, z, w, ρ))
        Lp_val_raw = sum(self.eval_Lp_raw(x, z, w, ρ))
        assert np.allclose(Lp_val_raw , Lp_val)
        return quad_xyw.partial(0, xD, np.array([]), z).argmin()

    def argmin_z(self, x, z, w, ρ):
        quad_xyw = self.Lp(w, ρ)
        xD = x.shape[0]
        zD = z.shape[0]
        Lp_val = sum(self.eval_Lp(x, z, w, ρ))
        Lp_val_raw = sum(self.eval_Lp_raw(x, z, w, ρ))
        assert np.allclose(Lp_val_raw , Lp_val)
        return quad_xyw.partial(xD, xD+zD, x, np.array([])).argmin()

    def solve_admm(self, ρ=1):
        xD = self.obj_x.l.shape[0]
        A = self.constraint.A[:, :xD]
        B = self.constraint.A[:, xD:]
        c = self.constraint.b
        x0 = self.obj_x.argmin()
        z0 = self.obj_z.argmin()
        return admm((self.argmin_x, self.argmin_z),
                    self.dual_feasibility_check,
                    (self.obj_x, self.obj_z),
                    (self.grad_f_x, self.grad_g_z),
                    x0, z0, A, B, c, ρ)

    def grad_f_x(self, x):
        return self.obj_x.grad()(x)

    def grad_g_z(self, z):
        return self.obj_z.grad()(z)

    def dual_feasibility_check(self, x, z, w, err):
        xD = self.obj_x.l.shape[0]
        assert np.allclose(self.constraint.A[:, :xD].T.dot(w) + self.grad_f_x(x), 0, atol=err)
        assert np.allclose(self.constraint.A[:, xD:].T.dot(w) + self.grad_g_z(z), 0, atol=err)

    def solve(self, test=True, ρ=1):
        """
        minimize xᵀQx + 2sᵗx + zᵀRz + 2uᵀz + wᵀ(Ax + Bz - c)

        Represent as a quadratic in xzw = [x, z, w]
        And then solve as lstsq
        """
        xD = self.obj_x.l.shape[0]
        zD = self.obj_z.l.shape[0]
        xzwopt = self.augmented_lagrangian(ρ).argmin()
        return xzwopt[:xD], xzwopt[xD:xD+zD]


def random_quadratic(xD = 1,
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
    LOG.info("{}".format( (Q, s, R, u, A, B, c)))
    return Q, s, R, u, A, B, c

def test_quadratic(example=random_quadratic):
    qadmm = QuadraticADMM(*example())
    xopt_admm, zopt_admm = qadmm.solve_admm()
    xopt, zopt = qadmm.solve()
    thresh = 1e-2
    assert np.linalg.norm(xopt - xopt_admm) < thresh
    assert np.linalg.norm(zopt - zopt_admm) < thresh

def custom_quadratic():
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
    test_custom_quadratic()
    #test_quadratic()


