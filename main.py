import numpy as np
from numpy.linalg import inv, norm

import data_receiver
from mathlib import *
from plotlib import *


class IMUTracker:

    def __init__(self, sampling, tinit=1, data_order={'w': 1, 'a': 2, 'm': 3}):
        '''
        @param sampling: sampling rate of the IMU, in Hz
        @param tinit: initialization time where the device is expected to be stay still, in second
        @param data_order: specify the order of data in the data array
        '''

        super().__init__()
        # ---- parameters ----
        self.sampling = sampling
        self.dt = 1 / sampling    # second
        self.tinit = tinit    # second
        self.data_order = data_order

        # ---- helpers ----
        idx = {1: [0, 3], 2: [3, 6], 3: [6, 9]}
        self._widx = idx[data_order['w']]
        self._aidx = idx[data_order['a']]
        self._midx = idx[data_order['m']]

    def attitudeTrack(self, data, cut1=10, cut2=10):
        '''
        Removes gravity from acceleration data and transform it into navitgaion frame.
        Also tracks device's orientation.
        
        @param data: (,9) ndarray
        @param cut1, cut2: cut the first and last few data to avoid potential corrupted data

        Return: (acc, orientation)
        '''

        # ------------------------------- #
        # ---- Initialization ----
        # ------------------------------- #

        # discard the first and last few readings
        # for some reason they fluctuate a lot
        w = data[cut1:-cut2, self._widx[0]:self._widx[1]]
        a = data[cut1:-cut2, self._aidx[0]:self._aidx[1]]
        m = data[cut1:-cut2, self._midx[0]:self._midx[1]]

        init_idx = int(self.tinit / self.dt)

        # ---- gravity ----
        gn = -a[:init_idx].mean(axis=0)
        gn = gn[:, np.newaxis]
        # save the initial magnitude of gravity
        g0 = np.linalg.norm(gn)

        # ---- magnetic field ----
        mn = m[:init_idx].mean(axis=0)
        # magnitude is not important
        mn = Normalized(mn)[:, np.newaxis]

        # ---- compute noise covariance ----
        avar = a[:init_idx].var(axis=0)
        wvar = w[:init_idx].var(axis=0)
        mvar = m[:init_idx].var(axis=0)
        print('acc var: ', avar, ', ', np.linalg.norm(avar))
        print('ang var: ', wvar, ', ', np.linalg.norm(wvar))
        print('mag var: ', mvar, ', ', np.linalg.norm(mvar))

        # ---- cut initialization data ----
        w = w[init_idx - 1:] - w[:init_idx].mean(axis=0)
        a = a[init_idx:]
        m = m[init_idx:]
        sample_number = np.shape(a)[0]

        # ---- define sensor noise ----
        gyro_noise = 10 * np.linalg.norm(wvar)
        acc_noise = 10 * np.linalg.norm(avar)
        mag_noise = 10 * np.linalg.norm(mvar)

        # ---- data container ----
        a_nav = []
        orientations = []

        # ---- states and covariance matrix ----
        P = 1e-10 * I(4)    # state covariance matrix
        q = np.array([[1, 0, 0, 0]]).T    # quaternion state
        init_ori = -gn / np.linalg.norm(gn)    # initial orientation

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
            mt = Normalized(m[t, np.newaxis].T)

            # ------------------------------- #
            # ---- 1. Propagation ----
            # ------------------------------- #

            Ft = F(q, wt, self.dt)
            Gt = G(q)
            Q = (gyro_noise * self.dt)**2 * Gt @ Gt.T

            q = Normalized(Ft @ q)
            P = Ft @ P @ Ft.T + Q

            # ------------------------------- #
            # ---- 2. Measurement Update ----
            # ------------------------------- #

            # Use normalized measurements to reduce error!

            # ---- acc and mag prediction ----
            pa = Normalized(-Rotate(q) @ gn)
            pm = Normalized(Rotate(q) @ mn)

            # ---- residual ----
            Eps = np.vstack((Normalized(at), mt)) - np.vstack((pa, pm))

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

            q = Normalized(q)
            P = 0.5 * (P + P.T)    # make sure P is symmertical

            # ------------------------------- #
            # ---- 4. other things ----
            # ------------------------------- #

            # ---- navigation frame acceleration ----
            conj = -I(4)
            conj[0, 0] = 1
            an = Rotate(conj @ q) @ at + gn

            # ---- navigation frame orientation ----
            orin = Rotate(conj @ q) @ init_ori

            # ---- saving data ----
            a_nav.append(an.T[0])
            orientations.append(orin.T[0])

            t += 1

        a_nav = np.array(a_nav)
        orientations = np.array(orientations)
        return (a_nav, orientations)

    def removeAccErr(self, a_nav, threshold=0.2, wn=(0.01, 15)):
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

        filtered_a_nav = Filt_signal([a_nav], dt=self.dt, wn=wn, btype='bandpass')[0]
        return filtered_a_nav

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
    data = receive_data('tcp')

    print('calculating...')
    a_nav, ori = tracker.attitudeTrack(data)
    a_nav_filtered = tracker.removeAccErr(a_nav)
    plot_3([a_nav, a_nav_filtered])

    v = tracker.zupt(a_nav_filtered, threshold=0.5)
    plot_3([v])

    p = tracker.positionTrack(a_nav_filtered, v)

    plot_3D([[p, 'position']])


plot_trajectory()
