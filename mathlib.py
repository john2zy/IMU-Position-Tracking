import numpy as np
from numpy.linalg import norm
import scipy.signal


def I(n):
    '''
    unit matrix
    just making its name prettier than np.eye
    '''
    return np.eye(n)


def normalized(x):
    try:
        return x / np.linalg.norm(x)
    except:
        return x


def skew(x):
    '''
    takes in a 3d column vector
    returns its Skew-symmetric matrix
    '''

    x = x.T[0]
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def rotate(q):
    '''
    rotation transformation matrix
    nav frame to body frame as q is expected to be q^nb
    R(q) @ x to rotate x
    '''

    qv = q[1:4, :]
    qc = q[0]
    return (qc**2 - qv.T @ qv) * I(3) - 2 * qc * skew(qv) + 2 * qv @ qv.T


def F(q, wt, dt):
    '''state transfer matrix'''

    w = wt.T[0]
    Omega = np.array([[0, -w[0], -w[1], -w[2]], [w[0], 0, w[2], -w[1]],
                      [w[1], -w[2], 0, w[0]], [w[2], w[1], -w[0], 0]])

    return I(4) + 0.5 * dt * Omega


def G(q):
    '''idk what its called '''

    q = q.T[0]
    return 0.5 * np.array([[-q[1], -q[2], -q[3]], [q[0], -q[3], q[2]],
                           [q[3], q[0], -q[1]], [-q[2], q[1], q[0]]])


def Hhelper(q, vector):
    # just for convenience
    x = vector.T[0][0]
    y = vector.T[0][1]
    z = vector.T[0][2]
    q0 = q.T[0][0]
    q1 = q.T[0][1]
    q2 = q.T[0][2]
    q3 = q.T[0][3]

    h = np.array([
        [q0*x - q3*y + q2*z, q1*x + q2*y + q3*z, -q2*x + q1*y + q0*z, -q3*x - q0*y + q1*z],
        [q3*x + q0*y - q1*z, q2*x - q1*y - q0*z, q1*x + q2*y + q3*z, q0*x - q3*y + q2*z],
        [-q2*x + q1*y +q0*z, q3*x + q0*y - q1*z, -q0*x + q3*y - q2*z, q1*x + q2*y + q3*z]
    ])
    return 2 * h


def H(q, gn, mn):
    '''
    Measurement matrix
    '''

    H1 = Hhelper(q, gn)
    H2 = Hhelper(q, mn)
    return np.vstack((-H1, H2))


def filtSignal(data, dt=0.01, wn=10, btype='lowpass', order=1):
    '''
    filter all data at once
    uses butterworth filter of scipy
    @param data: [...]
    @param dt: sampling time
    @param wn: critical frequency
    '''
    
    res = []
    n, s = scipy.signal.butter(order, wn, fs=1 / dt, btype=btype)
    for d in data:
        d = scipy.signal.filtfilt(n, s, d, axis=0)
        res.append(d)
    return res