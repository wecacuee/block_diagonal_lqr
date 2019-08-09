from abc import ABC, abstractmethod, abstractproperty
import doctest
from functools import partial
from logging import basicConfig, getLogger, DEBUG
basicConfig()
LOG = getLogger(__name__)
LOG.setLevel(DEBUG)

import numpy as np

class Func(ABC):
    """
    f: X ↦ Y
    """
    @abstractproperty
    def domain_size(self):
        """
        f: X ↦ Y

        If X = Rⁿ
        return n
        """
        raise NotImplementedError()

    @abstractproperty
    def shape(self):
        """
        f: X ↦ Y

        If Y = Rᵐ
        return m
        """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, x):
        """
        return f(x)
        """
        raise NotImplementedError()

    @abstractmethod
    def grad(self):
        """
        returns ∇ₓf
        """
        raise NotImplementedError()

    @abstractmethod
    def add(self, g):
        """
        Adds in the range while assuming same domain of f and g
        r(x) = f(x) + g(x)

        return r
        """
        raise NotImplementedError()

    __add__ = add


    @abstractmethod
    def add_concat(self, g):
        """
        Adds in the range while concatenates in domain of f and g

        r(x, z) = f(x) + g(z)

        return r
        """
        raise NotImplementedError()

    @abstractmethod
    def concat_concat(self, g):
        """
        Concatenates range and domain of f and g

        r(x, z) = [[f(x)],
                   [g(z)]]

        return r
        """
        raise NotImplementedError()

    __and__ = concat_concat

    @abstractmethod
    def mul(self, g):
        """
        Multiplies in the range while assuming same domain of f and g
        r(x) = f(x) * g(x)

        return r
        """
        raise NotImplementedError()

    __lmul__ = mul

    @abstractmethod
    def mul_concat(self, g):
        """
        Multiplies in the range while concatenates in domain of f and g

        r(x, z) = f(x) * g(z)

        return r
        """
        raise NotImplementedError()

    @abstractmethod
    def argmin(self):
        """
        return argminₓ f(x)
        """
        raise NotImplementedError()


    @abstractmethod
    def partial(self, vstart, vend, x0, w0):
        """
        For given x0, w0
        r_xw(z) = f([x0, z, w0])

        return r_xw
        """
        raise NotImplementedError()


class ScalarQuadFunc(Func):
    """
    f: x ↦ R
    """
    def __init__(self, Q, l, c):
        """
        """
        self._domain_size = xD = Q.shape[0]
        assert Q.shape == (xD, xD)
        assert l.shape == (xD, )
        self.Q = Q
        self.l = l
        self.c = c

    @property
    def domain_size(self):
        return self._domain_size

    @property
    def shape(self):
        return 1

    def __call__(self, x):
        """
        return f(x)
        """
        Q = self.Q
        l = self.l
        c = self.c
        return x.T.dot(Q).dot(x) + 2*l.T.dot(x) + c

    @classmethod
    def random(cls, xD=None):
        if xD is None:
            xD = np.random.randint(100)
        return cls(np.random.rand(xD, xD),
                   np.random.rand(xD),
                   np.random.rand(1))

    def grad(self):
        """
        return f'

        >>> f = ScalarQuadFunc.random()
        >>> x = np.random.rand(f.domain_size)
        >>> fx = f(x)
        >>> ε = 1e-12
        >>> gfx_numerical = np.zeros_like(x)
        >>> for i in range(x.shape[0]):
        ...     xpε = x.copy()
        ...     xpε[i] = x[i] + ε
        ...     gfx_numerical[i] = (f(xpε) - fx) / ε
        >>> gfx = f.grad()(x)
        >>> np.allclose(gfx, gfx_numerical, rtol=0.1)
        True
        """
        return AffineFunction((self.Q + self.Q.T), 2*self.l)

    def add(self, other):
        """
        r(x) = f(x) + g(x)

        return r

        >>> f = ScalarQuadFunc.random()
        >>> g = ScalarQuadFunc.random(xD=f.domain_size)
        >>> x = np.random.rand(f.domain_size)
        >>> np.allclose((f + g)(x), f(x) + g(x))
        True
        """
        Q = self.Q
        l = self.l
        c = self.c
        if isinstance(other, ScalarQuadFunc):
            assert self.Q.shape == other.Q.shape
            assert self.l.shape == other.l.shape
            Qo = other.Q
            lo = other.l
            co = other.c
        elif isinstance(other, AffineFunction):
            Qo = 0
            lo = 0.5 * other.A.reshape(-1)
            co = other.B
        else:
            raise NotImplementedError("Unknown type for {}".format(other))
        return ScalarQuadFunc(Q+Qo, l+lo, c+co)

    __add__ = add

    def __getitem__(self, i):
        if not isinstance(i, slice):
            i = slice(i, i+1, 1)
        return ScalarQuadFunc(self.Q[i, i], self.l[i], self.c)

    def add_concat(self, other):
        """
        r(x, z) = f(x) + g(z)

        return r

        >>> f = ScalarQuadFunc.random()
        >>> g = ScalarQuadFunc.random()
        >>> r = f.add_concat(g)
        >>> x = np.random.rand(f.domain_size)
        >>> z = np.random.rand(g.domain_size)
        >>> np.allclose(r(np.hstack((x, z))), f(x) + g(z))
        True
        """
        Q = self.Q
        l = self.l
        c = self.c
        Qo = other.Q
        lo = other.l
        co = other.c

        xD = self.domain_size
        oD = other.domain_size
        rD = xD + oD
        Qr = np.eye(rD)
        Qr[:xD, :xD] = Q
        Qr[xD:, xD:] = Qo
        lr = np.hstack((l, lo))
        cr = c + co
        return ScalarQuadFunc(Qr, lr, cr)

    def concat_concat(self, other):
        """
        r(x, z) = [f(x),
                   g(z)]

        return r
        """
        raise NotImplementedError(".. because it won't be scalar anymore")

    __and__ = concat_concat

    def argmin(self):
        """
         argmin_x f(x)

        >>> f = ScalarQuadFunc.random()
        >>> xopt = f.argmin()
        >>> x = np.random.rand(f.domain_size)
        >>> (f(x) >= f(xopt)).all()
        True
        """
        if isinstance(self.Q, np.ndarray) and isinstance(self.l, np.ndarray):
            return np.linalg.lstsq(self.Q, -self.l, rcond=None)[0]
        else:
            Q = self.Q
            l = self.l
            Qinv = np.linalg.pinv((Q + Q.T))
            return - 2 * Qinv * l

    def partial_f(self, vstart, vend):
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
        lr = AffineFunction(np.hstack((0.5*(Qzx + Qxz.T), 0.5*(Qzw + Qwz.T))), lz)
        cr = ScalarQuadFunc(np.vstack((np.hstack((Qxx, Qxw)),
                                       np.hstack((Qwx, Qww)))),
                            np.hstack((lx, lw)),
                            c)

        return ScalarQuadFunc(Qr, lr, cr)

    def partial(self, vstart, vend, x0, w0):
        """
        r_x0w0(z) = f([x0, z, w0])

        return r

        >>> xD = np.random.randint(100)
        >>> zD = np.random.randint(100)
        >>> wD = np.random.randint(100)
        >>> f = ScalarQuadFunc.random(xD=xD+zD+wD)
        >>> x0 = np.random.rand(xD)
        >>> w0 = np.random.rand(wD)
        >>> r = f.partial(xD, xD+zD, x0, w0)
        >>> z = np.random.rand(zD)
        >>> np.allclose(r(z), f(np.hstack((x0, z, w0))))
        True
        """
        ptial = self.partial_f(vstart, vend)
        x0w0 = np.hstack((x0, w0))
        Qr = ptial.Q
        lr = ptial.l(x0w0)
        cr = ptial.c(x0w0)
        return ScalarQuadFunc(Qr, lr, cr)

    def partial_argmin(self, vstart, vend):
        """
        r([x, w]) = argmin_z f([x, z, w])

        return r

        >>> xD = np.random.randint(100)
        >>> zD = np.random.randint(100)
        >>> wD = np.random.randint(100)
        >>> f = ScalarQuadFunc.random(xD=xD+zD+wD)
        >>> r = f.partial_argmin(xD, xD+zD)
        >>> x = np.random.rand(xD)
        >>> w = np.random.rand(wD)
        >>> z = np.random.rand(zD)
        >>> (f(np.hstack((x, z, w))) >= r(np.hstack((x, w)))).all()
        True
        """
        ptial = self.partial_f(vstart, vend)
        Q = ptial.Q
        l = ptial.l
        c = ptial.c
        Qinv = np.linalg.pinv(Q)
        return -Qinv * l

    __array_ufunc__ = None

    @classmethod
    def zero(cls, D):
        return cls(np.zeros((D,D)), np.zeros(D), 0)

    def mul(self, other):
        """
        Multiplies in the range while assuming same domain of f and g
        r(x) = f(x) * g(x)

        return r

        >>> f = ScalarQuadFunc.random()
        >>> α = np.random.rand()
        >>> r = f.mul(α)
        >>> x = np.random.rand(f.domain_size)
        >>> np.allclose(r(x), α * f(x))
        True
        """
        Q = self.Q
        l = self.l
        c = self.c
        if (isinstance(other, (float, int)) or
            (isinstance(other, np.ndarray) and other.size == 1)):
            return ScalarQuadFunc(other * Q, other * l, other * c)
        else:
            raise NotImplementedError(type(other))

    __lmul__ = mul

    def __rmul__(self, other):
        return self.__lmul__(other)

    def mul_concat(self, g):
        """
        Multiplies in the range while concatenates in domain of f and g

        r(x, z) = f(x) * g(z)

        return r
        """
        raise NotImplementedError("... because multiplication of two quadratics is 4th order poly")

    def __repr__(self):
        return "ScalarQuadFunc({}, {}, {})".format(self.Q, self.l, self.c)


class AffineFunction(Func):
    """
    f : X ↦ Y
    f(x) = AX + B
    """
    def __init__(self, A, B):
        self._domain_size = A.shape[-1]
        self._range_shape = B.shape
        self.A = A
        self.B = B

    def __call__(self, x):
        A = self.A
        B = self.B
        return A.dot(x) + B

    @property
    def domain_size(self):
        return self._domain_size

    @property
    def shape(self):
        return self._range_shape

    @property
    def T(self):
        return self

    def argmin(self):
        return np.max(np.abs(A), axis=0) * (-np.Inf)

    def grad(self):
        return A

    @classmethod
    def random(cls, yD=None, xD=None):
        if xD is None:
            xD = np.random.randint(100)
        if yD is None:
            yD = (np.random.randint(100),)

        return cls(np.random.rand(*yD, xD),
                   np.random.rand(*yD))

    def add_concat(self, other):
        """
        r(x, z) = f(x) + g(z)

        >>> f = AffineFunction.random()
        >>> g = AffineFunction.random(yD=f.shape)
        >>> r = f.add_concat(g)
        >>> x = np.random.rand(f.domain_size)
        >>> z = np.random.rand(g.domain_size)
        >>> np.allclose(r(np.hstack((x, z))), f(x) + g(z))
        True
        """
        return type(self)(np.hstack((self.A, other.A)), self.B + other.B)

    def add(self, other):
        """
        r(x) = f(x) + g(x)

        >>> f = AffineFunction.random()
        >>> g = AffineFunction.random(xD=f.domain_size, yD=f.shape)
        >>> r = f + g
        >>> x = np.random.rand(f.domain_size)
        >>> np.allclose(r(x), f(x) + g(x))
        True
        """
        return type(self)(self.A + other.A, self.B + other.B)

    __add__ = add

    def __getitem__(self, i):
        if not isinstance(i, slice):
            i = slice(i, i+1, 1)
        return type(self)(self.A[i, i], self.B[i])

    def concat_concat(self, other):
        """
        r(x, z) = [f(x),
                   g(z)]

        >>> f = AffineFunction.random()
        >>> g = AffineFunction.random()
        >>> r = f & g
        >>> x = np.random.rand(f.domain_size)
        >>> z = np.random.rand(g.domain_size)
        >>> np.allclose(r(np.hstack((x, z))), np.hstack((f(x) , g(z))))
        True
        """
        A = self.A
        B = self.B
        Ao = other.A
        bo = other.B

        xD = self.domain_size
        oxD = other.domain_size
        rxD = xD + oxD

        yD = self.shape[0]
        oyD = other.shape[0]
        ryD = yD + oyD

        Ar = np.zeros((ryD, rxD))
        Ar[:yD, ..., :xD] = A
        Ar[yD:, ...,  xD:] = Ao

        br = np.hstack((B.T, bo.T)).T
        return type(self)(Ar, br)

    __and__ = concat_concat

    def dot(self, other):
        """
        r(x) = f(x).T *g(x)

        >>> f = AffineFunction.random()
        >>> g = AffineFunction.random(xD=f.domain_size, yD=f.shape)
        >>> x = np.random.rand(f.domain_size)
        >>> np.allclose((f.dot(g))(x), f(x).T.dot(g(x)))
        True
        """
        A = self.A
        B = self.B
        if isinstance(other, np.ndarray):
            if other.ndim < 2:
                other = other.reshape(1, -1)
            return AffineFunction(other.dot(A), other.dot(B))
        Ao = other.A
        bo = other.B
        return ScalarQuadFunc(A.T.dot(Ao),
                              0.5*(B.T.dot(Ao) + bo.T.dot(A)),
                              B.T.dot(bo))

    mul = dot

    __array_ufunc__ = None
    def __rmul__(self, other):
        A = self.A
        B = self.B
        if isinstance(other, float):
            return type(self)(other * A, other * B)
        elif isinstance(other, np.ndarray):
            if not other.shape[-1] == self.shape[0]:
                raise ValueError("Bad shape {}. expected {}".format(
                    other.shape[-1], self.shape[0]))
            return type(self)(other.dot(A), other.dot(B))

    def __lmul__(self, other):
        if isinstance(other, float):
            return self.__lmul__(other)
        raise NotImplementedError()

    def mul_concat(self, other):
        """
        r(x,z) = f(x).T * g(z)

        >>> f = AffineFunction(np.array([[1, -1]]), np.array([0]))
        >>> g = AffineFunction(np.array([[1.]]), np.array([0]))
        >>> r = f.mul_concat(g)
        >>> isinstance(r, ScalarQuadFunc)
        True
        >>> r.Q
        array([[ 0. ,  0. ,  0.5],
               [ 0. ,  0. , -0.5],
               [ 0.5, -0.5,  0. ]])
        >>> r.l
        array([0., 0., 0.])
        >>> r.c
        0


        >>> f = AffineFunction.random()
        >>> g = AffineFunction.random(yD=f.shape)
        >>> r = f.mul_concat(g)
        >>> x = np.random.rand(f.domain_size)
        >>> z = np.random.rand(g.domain_size)
        >>> np.allclose(r(np.hstack((x, z))), f(x).T.dot(g(z)))
        True
        """
        A = self.A
        B = self.B
        Ao = other.A
        bo = other.B

        xD = A.shape[1]
        oD = Ao.shape[1]
        rD = xD + oD
        Qr = np.zeros((rD, rD))
        Qr[:xD, xD:] = 0.5 * A.T.dot(Ao)
        Qr[xD:, :xD] = 0.5 * Ao.T.dot(A)
        lr = 0.5 * np.hstack((bo.T.dot(A), B.T.dot(Ao)))
        cr = B.T.dot(bo)
        return ScalarQuadFunc(Qr, lr, cr)

    def partial(self, vstart, vend, before, after):
        """
        r_x0w0(z) = f(x0, z, w0)

        return r_x0w0

        >>> xD = np.random.randint(100)
        >>> zD = np.random.randint(100)
        >>> wD = np.random.randint(100)
        >>> f = AffineFunction.random(xD=xD+zD+wD)
        >>> x0 = np.random.rand(xD)
        >>> w0 = np.random.rand(wD)
        >>> r_x0w0 = f.partial(xD, xD+zD, x0, w0)
        >>> z = np.random.rand(zD)
        >>> np.allclose(r_x0w0(z), f(np.hstack((x0, z, w0))))
        True
        """
        keep = slice(vstart, vend)
        bef = slice(0, vstart)
        aft = slice(vend, None)
        A = self.A
        B = self.B
        Ar = A[:, keep]
        br = B + A[:, bef].dot(before) + A[:, aft].dot(after)
        return AffineFunction(Ar, br)

    def __repr__(self):
        return "AffineFunction({}, {})".format(self.A, self.B)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
