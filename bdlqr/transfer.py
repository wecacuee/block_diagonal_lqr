
from bdlqr.separable import SeparableLinearSystem, joint_linear_system


def fake_proximal_q_function(slsys, y, tilde_v, u):
    """
    Returns a proximal_q function for the slsys. Here we don't actually learn
    the q-function but derive it from the linear form.

    The proximal q-function is defined as the cost to go from the given y, ṽ, u
    V(y, ṽ) = arg min_v ∑ y Q y + 0.5 ρ |ṽ_t - v_t]_2^2
                y_t+1 = Ay y_t + Bv v_t

    V(y, ṽ) represents best value when v_t in the neighborhood of ṽ can be
    achieved.

    Q(y, ṽ, v) = y Q y + 0.5 ρ |ṽ - v|_2^2 + V(Ay y + Bv v, ṽ_{t+1})

    return Q
    """
    # V(y, ṽ) = arg min_u ∑ y Q y + 0.5 ρ |ṽ_t - v_t]_2^2
    #          y_t+2   = Ay y_{t+1}   + Bv v_t+1
    #          v_t+1   = E Ax v_t     + E Bu u_t
    Qy   = slsys.Qy
    Ay   = slsys.Ay
    Bv   = slsys.Bv
    QyT  = slsys.QyT
    E    = slsys.E
    P_new, o_new, K, k = affine_backpropagation(Qy, s, R, z, A, B, P, o)

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
