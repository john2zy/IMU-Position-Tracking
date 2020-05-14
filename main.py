# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# %%
# %matplotlib widget


# %%
import data_receiver
from mathlib import *


# %%
# sampling rate
DT = 0.05    # s
setDT(DT)
# the initialization interval
ts = 0.5    # s

# initial error estimation
sigma_P = 0.001    # initial angular error
sigma_Q = 0.001   # discrete time error
sigma_aR = 0.2    # measurement noise of acc
sigma_mR = 0.5    # measurement noise of mag

P = sigma_P * I(3)
Q = sigma_Q * I(3)
# aR = sigma_aR * I(3)
# mR = sigma_aR * I(3)
R0 = np.diag((sigma_aR, sigma_aR, sigma_aR, sigma_mR, sigma_mR, sigma_mR))

# %% [markdown]
# # data processing
# data order: gyroscorpe, accelerometer, magnetometer

# %%
r = data_receiver.Receiver()

data = []

for line in r.receive():
    data.append(line.split(','))

data = np.array(data, dtype = np.float)

# discard the first and last few readings
# for some reason they fluctuate a lot
w = data[5:-5, 0:3]
a = data[5:-5, 3:6]
m = data[5:-5, 6:9]

if(np.shape(w)[0] < ts/DT):
    print("not enough data for intialization!")

# %% [markdown]
# ## Initialization

# %%
w_bias = w[:int(ts/DT)].mean(axis = 0)

gn = a[:int(ts/DT)].mean(axis = 0)
gn = -np.array([gn]).T

mn = m[:int(ts/DT)].mean(axis = 0)
mn = mn / np.linalg.norm(mn)
mn = np.array([mn]).T

# cut the initialization data
w = w[int(ts/DT) - 1:] - w_bias
# w = w[int(ts/DT):]
a = a[int(ts/DT):]
m = m[int(ts/DT):]


# %%
plt.subplot(211)
plt.plot(w[:, 0], label='wx')
plt.plot(w[:, 1], label='wy')
plt.plot(w[:, 2], label='wz')
plt.legend()

plt.subplot(212)
plt.plot(a[:, 0], label='ax')
plt.plot(a[:, 1], label='ay')
plt.plot(a[:, 2], label='az')
plt.legend()


# %%
g_fig = plt.figure()
ax = g_fig.add_subplot(111, projection='3d')
# ax.plot(points[:, 0], points[:, 1], points[:, 1], 'o')

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
# ax.set_zlim(-1.5, 1.5)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.plot(gn[0], gn[1], gn[2], 'o')
ax.plot(mn[0], mn[1], mn[2], 'o')
ax.plot([0], [0], [0], 'ro')

print(gn.T)

# %% [markdown]
# ## Kalman Filter

# %%
g = []
orientation = []
qs = []

q = np.array([[1, 0, 0, 0]]).T

t = 0
while(t < np.shape(a)[0]):
    wt_1 = np.array([w[t]]).T  # w_t-1
    at = np.array([a[t]]).T
    mt = np.array([m[t]]).T

    # time update
    q = quad_mul(q, exp_q(0.5 * DT * wt_1))  # q_(t|t-1)

    Rtt_1 = R(q)  # R_(t|t-1)
    P = P + (DT * Rtt_1) @ Q @ (DT * Rtt_1).T  # P_(t|t-1)

    # measurement update
    Ht = Hessian(q, gn, mn)
    St = Ht @ P @ Ht.T + R0
    Kt = P @ Ht.T @ np.linalg.inv(St)
    Epsilon_t = np.vstack((at, mt)) - np.vstack((-Rtt_1 @ gn, Rtt_1 @ mn))

    Eta_t = Kt @ Epsilon_t  # Eta_t
    P = P - Kt @ St @ Kt.T  # P_(t|t)

    # Relinearize
    q = quad_mul(exp_q(0.5 * Eta_t), q)
    q = q / np.linalg.norm(q)

    # qs.append(q[1:4, :].T[0])
    g.append((R(q) @ gn).T[0])
    t += 1

g = np.array(g)
qs = np.array(qs)


# %%
q = np.array([[0, 0, 0, 1]]).T
p = np.array([[1, 0, 0]]).T

R(q) @ p

# %% [markdown]
# ### plotting results

# %%
plt.subplot(311)
plt.plot(g[:, 0], 'r-', label='gx')
plt.legend()
plt.ylim(-12, 12)

plt.subplot(312)
plt.plot(g[:, 1], 'g-', label='gy')
plt.legend()
plt.ylim(-12, 12)

plt.subplot(313)
plt.plot(g[:, 2], 'b-', label='gz')
plt.legend()
plt.ylim(-12, 12)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

N = 200
# ax.plot(orientation[:N, 0], orientation[:N, 1], orientation[:N, 2], 'bo')
ax.plot(g[:, 0], g[:, 1], g[:, 2], 'go')
# ax.plot(qs[:, 0], qs[:, 1], qs[:, 2], 'go')

# ax.set_xlim(-1.5, 1.5)
# ax.set_ylim(-1.5, 1.5)
# ax.set_zlim(-1.5, 1.5)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.plot([0], [0], [0], 'ro')


# %%


