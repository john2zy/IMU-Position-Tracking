import numpy as np
from numpy.linalg import norm

dt = 0


def setDT(DT):
    global dt
    dt = DT


def I(n):
    '''unit matrix'''
    return np.diag(list((1 for _ in range(n))))


def skew(x):
    '''
    takes in a 3d column vector
    returns its skew-symmetric matrix
    '''

    x = x.T[0]
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def quad_split(q):
    '''split q into vector part and real part'''
    return q[1:4, :], np.array([q[0, :]])


def R(q):
    qv, qc = quad_split(q)
    tmp1 = (qc**2 - qv.T @ qv) * I(3)
    tmp2 = -2 * qc * skew(qv)
    tmp3 = 2 * qv @ qv.T
    return tmp1 + tmp2 + tmp3


# def R(q):
#     '''
#     rotation transformation matrix
#     nav frame to body frame as q is expected to be q^nb
#     R(q) @ x to rotate x
#     '''

#     qv, qc = quad_split(q)

#     return qv @ qv.T + qc**2 * I(3) + 2 * qc * skew(qv) + skew(qv)**2


def Hessian(q, gn, mn):
    '''
    hessian matrix
    '''
    return np.vstack((-R(q) @ skew(gn), R(q) @ skew(mn)))


def exp_q(eta):
    '''R3 -> R4 mapping for linearization of quatronion representation'''
    eta_norm = norm(eta)
    return np.vstack((np.cos(eta_norm), eta / eta_norm * np.sin(eta_norm)))


def quad_mul(p, q):
    '''quatronion multiplication'''
    pv, pc = quad_split(p)

    pl1 = np.hstack((pc, -pv.T))
    pl2 = np.hstack((pv, pc * I(3) + skew(pv)))
    pl = np.vstack((pl1, pl2))
    return pl @ q


def F(q, at):
    '''state transfer matrix for position and velocity'''

    tmp = skew(R(q) @ at)
    r1 = np.hstack((I(3), dt * I(3), -0.5 * dt**2 * tmp))
    r2 = np.hstack((np.zeros((3, 1)), I(3), -dt * tmp))
    r3 = np.hstack((np.zeros((3, 1)), np.zeros((3, 3)), I(3)))

    return np.hstack((r1, r2, r3))


# q = np.array([[1, 0, 0, 0]]).T
# quad_mul(q, q)