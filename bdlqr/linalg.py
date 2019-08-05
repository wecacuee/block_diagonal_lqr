import doctest
from functools import partial
from logging import basicConfig, getLogger, DEBUG
basicConfig()
LOG = getLogger(__name__)
LOG.setLevel(DEBUG)

import numpy as np


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


    def partial(self, vstart, vend, x0, w0):
        """
        r_w0z0(z) = f(x0, z, w0)

        return r_w0z0
        """
        Q = self.Q
        l = self.l
        c = self.c
        s, e = vstart, vend
        zslice = slice(s, e)
        x0slice = slice(0, s)
        w0slice = slice(e, None)

        Qxx, Qxz, Qxw = Q[x0slice, x0slice], Q[x0slice, zslice], Q[x0slice, w0slice]
        Qzx, Qzz, Qzw = Q[ zslice, x0slice], Q[ zslice, zslice], Q[ zslice, w0slice]
        Qwx, Qwz, Qww = Q[w0slice, x0slice], Q[w0slice, zslice], Q[w0slice, w0slice]

        lx, lz, lw = l[x0slice], l[zslice], l[w0slice]

        # Solution
        # [x0, z, w0] [[Qxx, Qxz, Qxw] [x0] + 2 [lxᵀ, lzᵀ, lwᵀ] [x0] + c
        #              [Qzx, Qzz, Qzw] [z ]                     [z ]
        #              [Qwx, Qwz, Qww] [w0]                     [w0]
        #
        # = zᵀQzz z
        #    + 2 lzᵀ z + 2x0ᵀQxz z + 2w0ᵀQwz z
        #    + c + x0ᵀQxx x0 + w0ᵀ Qww w0 + x0ᵀ Qxw w0 + w0ᵀ Qwx x0 + 2 lxᵀx0 + 2 lwᵀw0
        Qr = Qzz
        lr = lz + Qxz.T.dot(x0) + Qwz.T.dot(w0)
        cr = (c
              + x0.T.dot(Qxx).dot(x0)
              + w0.T.dot(Qww).dot(w0)
              + x0.T.dot(Qxw).dot(w0)
              + w0.T.dot(Qwx).dot(x0)
              + 2*lx.T.dot(x0)
              + 2*lw.T.dot(0))
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

    def __repr__(self):
        return "QuadraticFunction({}, {}, {})".format(self.Q, self.l, self.c)

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

    def __repr__(self):
        return "AffineFunction({}, {})".format(self.A, self.b)

