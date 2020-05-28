import numpy as np
from numpy.linalg import inv, norm

import data_receiver
from mathlib import *
from plotlib import *


class IMUTracker:

    def __init__(self, sampling, data_order={'w': 1, 'a': 2, 'm': 3}):
        '''
        @param sampling: sampling rate of the IMU, in Hz
        @param tinit: initialization time where the device is expected to be stay still, in second
        @param data_order: specify the order of data in the data array
        '''

        super().__init__()
        # ---- parameters ----
        self.sampling = sampling
        self.dt = 1 / sampling    # second
        self.data_order = data_order

        # ---- helpers ----
        idx = {1: [0, 3], 2: [3, 6], 3: [6, 9]}
        self._widx = idx[data_order['w']]
        self._aidx = idx[data_order['a']]
        self._midx = idx[data_order['m']]

    def initialize(self, data, noise_coefficient={'w': 100, 'a': 100, 'm': 10}):
        '''
        Algorithm initialization
        
        @param data: (,9) ndarray
        @param cut: cut the first few data to avoid potential corrupted data
        @param noise_coefficient: sensor noise is determined by variance magnitude times this coefficient
        
        Return: a list of initialization values used by EKF algorithm: 
        (gn, g0, mn, gyro_noise, gyro_bias, acc_noise, mag_noise)
        '''

        # discard the first few readings
        # for some reason they might fluctuate a lot
        w = data[:, self._widx[0]:self._widx[1]]
        a = data[:, self._aidx[0]:self._aidx[1]]
        m = data[:, self._midx[0]:self._midx[1]]

        # ---- gravity ----
        gn = -a.mean(axis=0)
        gn = gn[:, np.newaxis]
        # save the initial magnitude of gravity
        g0 = np.linalg.norm(gn)

        # ---- magnetic field ----
        mn = m.mean(axis=0)
        # magnitude is not important
        mn = normalized(mn)[:, np.newaxis]

        # ---- compute noise covariance ----
        avar = a.var(axis=0)
        wvar = w.var(axis=0)
        mvar = m.var(axis=0)
        print('acc var: %s, norm: %s' % (avar, np.linalg.norm(avar)))
        print('ang var: %s, norm: %s' % (wvar, np.linalg.norm(wvar)))
        print('mag var: %s, norm: %s' % (mvar, np.linalg.norm(mvar)))

        # ---- define sensor noise ----
        gyro_noise = noise_coefficient['w'] * np.linalg.norm(wvar)
        gyro_bias = w.mean(axis=0)
        acc_noise = noise_coefficient['a'] * np.linalg.norm(avar)
        mag_noise = noise_coefficient['m'] * np.linalg.norm(mvar)
        return (gn, g0, mn, gyro_noise, gyro_bias, acc_noise, mag_noise)

    def attitudeTrack(self, data, init_list):
        '''
        Removes gravity from acceleration data and transform it into navitgaion frame.
        Also tracks device's orientation.
        
        @param data: (,9) ndarray
        @param list: initialization values for EKF algorithm: 
        (gn, g0, mn, gyro_noise, gyro_bias, acc_noise, mag_noise)

        Return: (acc, orientation)
        '''

        # ------------------------------- #
        # ---- Initialization ----
        # ------------------------------- #
        gn, g0, mn, gyro_noise, gyro_bias, acc_noise, mag_noise = init_list
        w = data[:, self._widx[0]:self._widx[1]] - gyro_bias
        a = data[:, self._aidx[0]:self._aidx[1]]
        m = data[:, self._midx[0]:self._midx[1]]
        sample_number = np.shape(data)[0]

        # ---- data container ----
        a_nav = []
        orix = []
        oriy = []
        oriz = []

        # ---- states and covariance matrix ----
        P = 1e-10 * I(4)    # state covariance matrix
        q = np.array([[1, 0, 0, 0]]).T    # quaternion state
        init_ori = I(3)   # initial orientation

        # ------------------------------- #
        # ---- Extended Kalman Filter ----
        # ------------------------------- #

        # all vectors are column vectors

        t = 0
        while t < sample_number:

            # ------------------------------- #
            # ---- 0. Data Preparation ----
            # ------------------------------- #

            wt = w[t, np.newaxis].T
            at = a[t, np.newaxis].T
            mt = normalized(m[t, np.newaxis].T)

            # ------------------------------- #
            # ---- 1. Propagation ----
            # ------------------------------- #

            Ft = F(q, wt, self.dt)
            Gt = G(q)
            Q = (gyro_noise * self.dt)**2 * Gt @ Gt.T

            q = normalized(Ft @ q)
            P = Ft @ P @ Ft.T + Q

            # ------------------------------- #
            # ---- 2. Measurement Update ----
            # ------------------------------- #

            # Use normalized measurements to reduce error!

            # ---- acc and mag prediction ----
            pa = normalized(-rotate(q) @ gn)
            pm = normalized(rotate(q) @ mn)

            # ---- residual ----
            Eps = np.vstack((normalized(at), mt)) - np.vstack((pa, pm))

            # ---- sensor noise ----
            # R = internal error + external error
            Ra = [(acc_noise / np.linalg.norm(at))**2 + (1 - g0 / np.linalg.norm(at))**2] * 3
            Rm = [mag_noise**2] * 3
            R = np.diag(Ra + Rm)

            # ---- kalman gain ----
            Ht = H(q, gn, mn)
            S = Ht @ P @ Ht.T + R
            K = P @ Ht.T @ np.linalg.inv(S)

            # ---- actual update ----
            q = q + K @ Eps
            P = P - K @ Ht @ P

            # ------------------------------- #
            # ---- 3. Post Correction ----
            # ------------------------------- #

            q = normalized(q)
            P = 0.5 * (P + P.T)    # make sure P is symmertical

            # ------------------------------- #
            # ---- 4. other things ----
            # ------------------------------- #

            # ---- navigation frame acceleration ----
            conj = -I(4)
            conj[0, 0] = 1
            an = rotate(conj @ q) @ at + gn

            # ---- navigation frame orientation ----
            orin = rotate(conj @ q) @ init_ori

            # ---- saving data ----
            a_nav.append(an.T[0])
            orix.append(orin.T[0, :])
            oriy.append(orin.T[1, :])
            oriz.append(orin.T[2, :])

            t += 1

        a_nav = np.array(a_nav)
        orix = np.array(orix)
        oriy = np.array(oriy)
        oriz = np.array(oriz)
        return (a_nav, orix, oriy, oriz)

    def removeAccErr(self, a_nav, threshold=0.2, filter=False, wn=(0.01, 15)):
        '''
        Removes drift in acc data assuming that
        the device stays still during initialization and ending period.
        The initial and final acc are inferred to be exactly 0.
        The final acc data output is passed through a bandpass filter to further reduce noise and drift.
        
        @param a_nav: acc data, raw output from the kalman filter
        @param threshold: acc threshold to detect the starting and ending point of motion
        @param wn: bandpass filter cutoff frequencies
        
        Return: corrected and filtered acc data
        '''

        sample_number = np.shape(a_nav)[0]
        t_start = 0
        for t in range(sample_number):
            at = a_nav[t]
            if np.linalg.norm(at) > threshold:
                t_start = t
                break

        t_end = 0
        for t in range(sample_number - 1, -1, -1):
            at = a_nav[t]
            if np.linalg.norm(at - a_nav[-1]) > threshold:
                t_end = t
                break

        an_drift = a_nav[t_end:].mean(axis=0)
        an_drift_rate = an_drift / (t_end - t_start)

        for i in range(t_end - t_start):
            a_nav[t_start + i] -= (i + 1) * an_drift_rate

        for i in range(sample_number - t_end):
            a_nav[t_end + i] -= an_drift

        if filter:
            filtered_a_nav = filtSignal([a_nav], dt=self.dt, wn=wn, btype='bandpass')[0]
            return filtered_a_nav
        else:
            return a_nav

    def zupt(self, a_nav, threshold):
        '''
        Applies Zero Velocity Update(ZUPT) algorithm to acc data.
        
        @param a_nav: acc data
        @param threshold: stationary detection threshold, the more intense the movement is the higher this should be

        Return: velocity data
        '''

        sample_number = np.shape(a_nav)[0]
        velocities = []
        prevt = -1
        still_phase = False

        v = np.zeros((3, 1))
        t = 0
        while t < sample_number:
            at = a_nav[t, np.newaxis].T

            if np.linalg.norm(at) < threshold:
                if not still_phase:
                    predict_v = v + at * self.dt

                    v_drift_rate = predict_v / (t - prevt)
                    for i in range(t - prevt - 1):
                        velocities[prevt + 1 + i] -= (i + 1) * v_drift_rate.T[0]

                v = np.zeros((3, 1))
                prevt = t
                still_phase = True
            else:
                v = v + at * self.dt
                still_phase = False

            velocities.append(v.T[0])
            t += 1

        velocities = np.array(velocities)
        return velocities

    def positionTrack(self, a_nav, velocities):
        '''
        Simple integration of acc data and velocity data.
        
        @param a_nav: acc data
        @param velocities: velocity data
        
        Return: 3D coordinates in navigation frame
        '''

        sample_number = np.shape(a_nav)[0]
        positions = []
        p = np.array([[0, 0, 0]]).T

        t = 0
        while t < sample_number:
            at = a_nav[t, np.newaxis].T
            vt = velocities[t, np.newaxis].T

            p = p + vt * self.dt + 0.5 * at * self.dt**2
            positions.append(p.T[0])
            t += 1

        positions = np.array(positions)
        return positions


def receive_data(mode='tcp'):
    data = []

    if mode == 'tcp':
        r = data_receiver.Receiver()
        file = open('data.txt', 'w')
        print('listening...')
        for line in r.receive():
            file.write(line)
            data.append(line.split(','))
        data = np.array(data, dtype=np.float)
        return data

    if mode == 'file':
        file = open('data.txt', 'r')
        for line in file.readlines():
            data.append(line.split(','))
        data = np.array(data, dtype=np.float)
        return data

    else:
        raise Exception('Invalid mode argument: ', mode)


def plot_trajectory():
    tracker = IMUTracker(sampling=100)
    data = receive_data('file')    # toggle data source between 'tcp' and 'file' here

    print('initializing...')
    init_list = tracker.initialize(data[5:30])

    print('--------')
    print('processing...')
    
    # EKF step
    a_nav, orix, oriy, oriz = tracker.attitudeTrack(data[30:], init_list)

    # Acceleration correction step
    a_nav_filtered = tracker.removeAccErr(a_nav, filter=False)
    # plot3([a_nav, a_nav_filtered])

    # ZUPT step
    v = tracker.zupt(a_nav_filtered, threshold=0.2)
    # plot3([v])

    # Integration Step
    p = tracker.positionTrack(a_nav_filtered, v)
    plot3D([[p, 'position']])
    
    # make 3D animation
    # xl = np.min(p[:, 0]) - 0.05
    # xh = np.max(p[:, 0]) + 0.05
    # yl = np.min(p[:, 1]) - 0.05
    # yh = np.max(p[:, 1]) + 0.05
    # zl = np.min(p[:, 2]) - 0.05
    # zh = np.max(p[:, 2]) + 0.05
    # plot3DAnimated(p, lim=[[xl, xh], [yl, yh], [zl, zh]], label='position', interval=5)


if __name__ == '__main__':
    plot_trajectory()
