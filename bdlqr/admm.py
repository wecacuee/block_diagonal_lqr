import numpy as np

def argminLp_z(x, z0, w):
    zopt = x0
    return zopt

def argminLp_x(x0, z, w):
    xopt = x0
    return xopt

def admm(argmin_Lp, x0, z0, A, B, c, ρ, max_iter=100, thresh=1e-4):
    """
    minimize f(x) + g(z)
        s.t. Ax + Bz = c
    """
    xk = x0
    zk = z0
    wk = np.zeros_like(c)
    for k in range(max_iter):
        xkp1 = argmin_Lp(xk, zk, wk, ρ, "x")
        zkp1 = argmin_Lp(xkp1, zk, wk, ρ, "z")
        wk = wk + ρ*(A.dot(xkp1) + B.dot(zkp1) - c)
        if (np.linalg.norm(xk - xkp1)
            + np.linalg.norm(zk - zkp1)) < thresh:
            break
        xk = xkp1
        zk = zkp1
    return xk, zk


class QuadraticADMM:
    """
    minimize xᵀQx + 2sᵗx + zᵀRz + 2uᵀz
    s.t. Ax + Bz = c
    """
    def __init__(self, Q, s, R, u, A, B, c):
        self.Q = Q
        self.s = s
        self.R = R
        self.u = u
        self.A = A
        self.B = B
        self.c = c
        self.minimizers = dict(x=self.argmin_x,
                               z=self.argmin_z)

    def Lp(self, w, ρ):
        """
        represent f(x) + g(z) + wᵀ(Ax+Bz-c) + 0.5ρ|Ax+Bz-c|₂²
        as quadratic in x, z
        xᵀLQx x + 2*zᵀ LQxz x + zᵀ LQz z + 2 lxᵀ x + 2 lzᵀ z

        Return LQx, LQxz, LQxz, lx, lz and ls
        """
        Q = self.Q
        s = self.s
        R = self.R
        u = self.u
        A = self.A
        B = self.B
        c = self.c
        # Solution
        # xᵀQx + 2sᵗx + zᵀRz + 2uᵀz + wᵀ(Ax+Bz-c) + 0.5ρ|Ax+Bz-c|₂²
        # = xᵀ(Q+AᵀA)x + (2sᵀ +  wᵀA - ρcᵀA)x
        #   + zᵀ(R+BᵀB)z + (2uᵀ + wᵀB - ρcᵀB)z
        #   + ρzᵀBᵀAx - wᵀc + ρcᵀc
        LQx = Q + A.T.dot(A)
        LQxz = 0.5*ρ*B.T.dot(A)
        LQz = R+B.T.dot(B)
        lx = s + 0.5*A.T.dot(w) - 0.5*ρ*A.T.dot(c)
        lz = u + 0.5*B.T.dot(w) - 0.5*ρ*B.T.dot(c)
        return LQx, LQxz, LQz, lx, lz

    def argmin_x(self, _, z, w, ρ):
        LQx, LQxz, _, lx, _ = self.Lp(w, ρ)
        xopt, _, _, _ = np.linalg.lstsq(LQx, - LQxz.T.dot(z) - lx)
        return xopt

    def argmin_z(self, x, z, w, ρ):
        _, LQxz, LQz, _, lz = self.Lp(w, ρ)
        zopt, _, _, _ = np.linalg.lstsq(LQz, - LQxz.dot(x) - lz)
        return zopt

    def argmin(self, x, z, w, ρ, which_var):
        return self.minimizers[which_var](x, z, w, ρ)

    def solve_admm(self, ρ=1):
        Q = self.Q
        s = self.s
        R = self.R
        u = self.u
        A = self.A
        B = self.B
        c = self.c
        x0 = np.zeros_like(s)
        z0 = np.zeros_like(u)
        w0 = np.zeros_like(c)
        return admm(self.argmin, x0, z0, A, B, c, ρ)

    def solve(self):
        """
        minimize xᵀQx + 2sᵗx + zᵀRz + 2uᵀz + wᵀ(Ax + Bz - c)

        Represent as a quadratic in xzw = [x, z, w]
        And then solve as lstsq
        """
        Q = self.Q
        s = self.s
        R = self.R
        u = self.u
        A = self.A
        B = self.B
        c = self.c
        # Solution:
        # Q = [[   Q,    0, 0.5 Aᵀ],
        #      [   0,    R, 0.5 Bᵀ],
        #      [0.5A, 0.5B,      0]]
        #
        # l = [s, u, -c]
        z_QR = np.zeros((Q.shape[0], R.shape[1]))
        z_BB = np.zeros((B.shape[0], B.shape[0]))
        Q_xzw = np.vstack([np.hstack([     Q,    z_QR, 0.5 * A.T]),
                           np.hstack([z_QR.T,       R, 0.5 * B.T]),
                           np.hstack([ 0.5*A,   0.5*B,      z_BB])])
        l_xzw = np.hstack([s, u, -0.5*c])
        xzwopt, _, _, _ = np.linalg.lstsq(Q_xzw, - l_xzw)
        xD = s.shape[0]
        zD = u.shape[0]
        return xzwopt[:xD], xzwopt[xD:xD+zD]

def random_quadratic():
    xD = 3
    zD = 2
    cD = 1
    Qsqrt = np.random.rand(xD,xD)
    Q = Qsqrt.T.dot(Qsqrt)
    s = np.random.rand(xD)
    Rsqrt = np.random.rand(zD, zD)
    R = Rsqrt.T.dot(Rsqrt)
    u = np.random.rand(zD)
    A = np.random.rand(cD, xD)
    B = np.random.rand(cD, zD)
    c = np.random.rand(cD)
    return Q, s, R, u, A, B, c

if __name__ == '__main__':
    qadmm = QuadraticADMM(*random_quadratic())
    xopt_admm, zopt_admm = qadmm.solve_admm()
    xopt, zopt = qadmm.solve()
    thresh = 1e-2
    assert np.linalg.norm(xopt - xopt_admm) < thresh
    assert np.linalg.norm(zopt - zopt_admm) < thresh


