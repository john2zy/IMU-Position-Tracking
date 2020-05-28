import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# plt.style.use('fivethirtyeight')


def plotSignal(al: list, wl: list, ml: list):
    f, ax = plt.subplots(ncols=3, nrows=3)
    plot3(al, ax=ax[:, 0])
    plot3(wl, ax=ax[:, 1])
    plot3(ml, ax=ax[:, 2])
    ax[0, 0].set_ylabel('x')
    ax[1, 0].set_ylabel('y')
    ax[2, 0].set_ylabel('z')
    ax[2, 0].set_xlabel('a')
    ax[2, 1].set_xlabel(r'$\omega$')
    ax[2, 2].set_xlabel('m')


def plotgAndAcc(g, ab):
    '''
    plot tracked gravity and body frame acceleration
    '''

    fig, ax = plt.subplots(nrows=1, ncols=3)
    plot3([g, ab],
          ax=ax,
          lims=[[None, [-12, 12]]] * 3,
          labels=[['$g_x$', '$g_y$', '$g_z$'], ['$a^b_x$', '$a^b_y$', '$a^b_z$']],
          show_legend=True)


def plot3(data, ax=None, lims=None, labels=None, show=False, show_legend=False):
    '''
    @param data: [ndarray, ...]
    @param lims: [[[xl, xh], [yl, yh]], ...]
    @param labels: [[label_string, ...], ...]
    '''

    show_flag = False
    if ax is None:
        show_flag = True
        f, ax = plt.subplots(ncols=1, nrows=3)

    for axel in range(3):
        has_label = False
        for n in range(len(data)):
            d = data[n]
            label = labels[n] if labels is not None else None

            if label is not None:
                ax[axel].plot(d[:, axel], label=label[axel])
                has_label = True
            else:
                ax[axel].plot(d[:, axel])

            lim = lims[axel] if lims is not None else None
            if lim is not None:
                if lim[0] is not None:
                    ax[axel].set_xlim(lim[0][0], lim[0][1])
                if lim[1] is not None:
                    ax[axel].set_ylim(lim[1][0], lim[1][1])

        if (has_label is not None) and show_legend:
            ax[axel].legend()
        ax[axel].grid(True)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    if show or show_flag:
        plt.show()
    return ax


def plot3D(data, lim=None, ax=None):
    '''
    @param data: [[data, label_string], ...]
    @param lim: [[xl, xh], [yl, yh], [zl, zh]]
    '''

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for item in data:
        label = item[1]
        d = item[0]
        ax.plot(d[:, 0], d[:, 1], d[:, 2], 'o', label=label)

    if lim is not None:
        if lim[0] is not None:
            ax.set_xlim(lim[0][0], lim[0][1])
        if lim[1] is not None:
            ax.set_ylim(lim[1][0], lim[1][1])
        if lim[2] is not None:
            ax.set_zlim(lim[2][0], lim[2][1])

    ax.legend()
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.plot([0], [0], [0], 'ro')
    plt.show()


# ----------------------------- #
# ---- Animated Plots ----
# ----------------------------- #


def plot3DAnimated(data, lim=[[-1, 1], [-1, 1], [-1, 1]], label=None, interval=10, show=True, repeat=False):
    '''
    @param data: (n, 3) ndarray
    @param lim: [[xl, xh], [yl, yh], [zl, zh]]
    @param show: if it's set to false, you can call this function multiple times to draw multiple lines
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if label is not None:
        ln, = ax.plot([], [], [], 'o', label=label)
    else:
        ln, = ax.plot([], [], [], 'o')

    def init():
        ax.plot([0], [0], [0], 'ro')
        ax.set_xlim(lim[0][0], lim[0][1])
        ax.set_ylim(lim[1][0], lim[1][1])
        ax.set_zlim(lim[2][0], lim[2][1])
        if label is not None:
            ax.legend()
        return ln,

    def update(frame):
        ln.set_xdata(data[:frame, 0])
        ln.set_ydata(data[:frame, 1])
        ln.set_3d_properties(data[:frame, 2])
        return ln,

    ani = FuncAnimation(fig,
                        update,
                        frames=range(1,
                                     np.shape(data)[0] + 1),
                        init_func=init,
                        blit=True,
                        interval=interval,
                        repeat=repeat)
    if show:
        plt.show()