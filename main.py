import numpy as np
import numpy.linalg

import data_receiver
from mathlib import *
from plotlib import *
from butter import *


class PositionEstimator:

    def __init__(self,
                 sampling,
                 tinit=0.5,
                 gyro_noise=2e-4,
                 acc_noise=2e-3,
                 mag_noise=2e-1,
                 filtercutoff=10,
                 order={
                     'w': 1,
                     'a': 2,
                     'm': 3
                 }):
        '''
        @param sampling: sampling rate of the IMU, in Hz
        @param tinit: initialization time where the device is expected to be stay still, in second
        @param gyro_noise: gyroscope measurement noise, it's expected to be low
        @param acc_noise: accelerometer measurement noise, it should be higher than that of gyroscope noise
        @param mag_noise: magnetometer measurement noise, it should be the highest among the three
        @param filtercutoff: cutoff frequency for the lowpass filter, in Hz
        @param order: specify the order of data in the data array
        '''

        super().__init__()
        # parameters
        self.sampling = sampling
        self.dt = 1 / sampling    # second
        self.tinit = tinit    # second
        self.gyro_noise = gyro_noise
        self.acc_noise = acc_noise
        self.mag_noise = mag_noise
        self.filtercutoff = filtercutoff
        self.order = order

        # helpers
        idx = {1: [0, 3], 2: [3, 6], 3: [6, 9]}
        self.widx = idx[self.order['w']]
        self.aidx = idx[self.order['a']]
        self.midx = idx[self.order['m']]
        self.initialized = False    # flag

        # state variables
        self.P = 1e-8 * I(4)    # state covariance matrix
        self.q = np.array([[1, 0, 0, 0]]).T    # quaternion state
        self.p = np.array([[0, 0, 0]]).T    # position stae
        self.v = np.array([[0, 0, 0]]).T    # velocity state

        # orientation state: [x, y, z] basis in colunm vectors.
        # this assumes the initial body frame is navigation frame.
        self.ori = I(3)

        # data recordings
        self.a_filtered = []
        self.w_filtered = []
        self.m_filtered = []
        self.positions = []    # need to add recording in send()
        self.orientations = []    # need to add recording in send()
        self.g = []    # body frame gravity estimations
        self.abody = []    # body frame acceleration estimations

        # filters
        self.filters = list((Butter(btype='lowpass',
                                    cutoff=self.filtercutoff,
                                    sampling=self.sampling) for _ in range(9)))

    def initialize(self, data, cut1=10, cut2=10):
        '''
        @param data: (n, 9) ndarray
        @param cut1, cut2: the number of the first and last few readings to discard, becuase they are sometimes inaccurate
        '''

        # discard the first and last few readings
        # for some reason they fluctuate a lot
        w = data[cut1:-cut2, self.widx[0]:self.widx[1]]
        a = data[cut1:-cut2, self.aidx[0]:self.aidx[1]]
        m = data[cut1:-cut2, self.midx[0]:self.midx[1]]

        if (np.shape(w)[0] < self.tinit // self.dt):
            raise Exception("not enough data for intialization")

        # ---- gravity ----
        self.gn = -a.mean(axis=0)
        self.gn = self.gn[:, np.newaxis]
        # save the initial magnitude of gravity
        self.g0 = np.linalg.norm(self.gn)

        # ---- magnetic field ----
        self.mn = m.mean(axis=0)
        # magnitude is not important
        self.mn = Normalized(self.mn)[:, np.newaxis]

        # ---- compute noise covariance ----
        # self.wvar = w.var(axis=0)
        # self.avar = a.var(axis=0)
        # self.mvar = m.var(axis=0)
        # print("angular variance: ", self.wvar)
        # print("acc variance: ", self.avar)
        # print("mag variance: ", self.mvar)
        # if self.wvar > 0.5 or self.avar > 0.5 or self.mvar > 0.5:
        #     raise Exception(
        #         "variance is too high, please don't move the device during initialization"
        #     )

        # ---- initialize filters as well ----
        for i in range(3):
            # send() receives a list and returns a list
            self.filters[i].send(w[:, i].tolist())
            self.filters[i + 3].send(a[:, i].tolist())
            self.filters[i + 6].send(m[:, i].tolist())

        self.initialized = True

    def send(self, data):
        '''
        @param data: (,9) ndarray
        '''

        if not self.initialized:
            raise Exception("please initialize first")

        w = data[self.widx[0]:self.widx[1]]
        a = data[self.aidx[0]:self.aidx[1]]
        m = data[self.midx[0]:self.midx[1]]

        # ---- filter data ----
        w_filtered = []
        a_filtered = []
        m_filtered = []
        for i in range(3):
            # send() receives a list and returns a list
            w_filtered.append(self.filters[i].send([w[i]])[0])
            a_filtered.append(self.filters[i + 3].send([a[i]])[0])
            m_filtered.append(self.filters[i + 6].send([m[i]])[0])

        self.a_filtered.append(a_filtered)
        self.w_filtered.append(w_filtered)
        self.m_filtered.append(m_filtered)

        # ------------------------------- #
        # ---- Extended Kalman Filter ----
        # ------------------------------- #

        # all vectors are column vectors

        # ------------------------------- #
        # ---- 0. Data Preparation ----
        # ------------------------------- #

        # -- uncomment to use unfiltered data -- #
        # wt = w[:, np.newaxis]
        # at = a[:, np.newaxis]
        # mt = Normalized(m[:, np.newaxis])

        wt = np.array([w_filtered]).T
        at = np.array([a_filtered]).T
        mt = Normalized(np.array([m_filtered]).T)

        # ------------------------------- #
        # ---- 1. Propagation ----
        # ------------------------------- #

        Ft = F(self.q, wt, self.dt)
        Gt = G(self.q)
        Q = (self.gyro_noise * self.dt)**2 * Gt @ Gt.T

        self.q = Normalized(Ft @ self.q)
        self.P = Ft @ self.P @ Ft.T + Q

        # ------------------------------- #
        # ---- 2. Measurement Update ----
        # ------------------------------- #

        # Use normalized measurements to reduce error!

        # ---- acc and mag prediction ----
        pa = Normalized(-Rotate(self.q) @ self.gn)
        pm = Normalized(Rotate(self.q) @ self.mn)

        # ---- Residual ----
        Eps = np.vstack((Normalized(at), mt)) - np.vstack((pa, pm))

        # ---- sensor noise ----
        # R = internal error + external error
        Ra = [(self.acc_noise / np.linalg.norm(at))**2 +
              (1 - self.g0 / np.linalg.norm(at))**2] * 3
        Rm = [self.mag_noise**2] * 3
        R = np.diag(Ra + Rm)

        # ---- Kalman Gain ----
        Ht = H(self.q, self.gn, self.mn)
        S = Ht @ self.P @ Ht.T + R
        K = self.P @ Ht.T @ np.linalg.inv(S)

        # ---- actual update ----
        self.q = self.q + K @ Eps
        self.P = self.P - K @ Ht @ self.P

        # ------------------------------- #
        # ---- 3. Post Correction ----
        # ------------------------------- #

        self.q = Normalized(self.q)
        self.P = 0.5 * (self.P + self.P.T)    # make sure P is symmertical

        # ------------------------------- #
        # ---- 4. other things ----
        # ------------------------------- #

        # ---- gravity ----
        gt = Rotate(self.q) @ self.gn
        gt = self.g0 * Normalized(gt)
        self.g.append(gt.T[0])

        # ---- body frame velocity ----
        ab = at + gt
        self.abody.append(ab.T[0])
        self.v = self.v + ab * self.dt
        self.p = self.p + self.v * self.dt + 0.5 * ab * self.dt**2
        # self.positions.append(...)

        # ---- orientation ----
        # conj is used to get a conjugate quaternion,
        # yes I know, this is terrible
        conj = -I(4)
        conj[0, 0] = 1
        qc = conj @ self.q
        self.ori = Rotate(qc) @ self.ori
        # renormalize orientation
        for i in range(3):
            self.ori[:, i] = self.ori[:, i] / np.linalg.norm(self.ori[:, i])
        # self.orientations.append(...)

        return self.p.T[0], self.ori


r = data_receiver.Receiver()
estimator = PositionEstimator(100)

d = []
positions = []
initdata = []
count = 0

print('Listening...')
for line in r.receive():
    data = line.split(',')
    if 0 <= count < 100:
        initdata.append(data)
        count += 1
        continue
    elif count >= 100:
        initdata = np.array(initdata, dtype=np.float)
        estimator.initialize(initdata)
        print("Initialization complete...")
        count = -1
        continue

    d.append(data)
    data = np.array(data, dtype=np.float)

    position, orientation = estimator.send(data)
    positions.append(position)

d = np.array(d, dtype=np.float)
plot_signal([d[:, 3:6]], [d[:, 0:3]], [d[:, 6:9]])

positions = np.array(positions, dtype=np.float)
plot_3D([[positions, "position"]])

# estimator.a_filtered = np.array(estimator.a_filtered, dtype=np.float)
# estimator.w_filtered = np.array(estimator.w_filtered, dtype=np.float)
# estimator.m_filtered = np.array(estimator.m_filtered, dtype=np.float)

estimator.g = np.array(estimator.g, dtype=np.float)
estimator.abody = np.array(estimator.abody, dtype=np.float)
plot_g_and_acc(estimator.g, estimator.abody)
