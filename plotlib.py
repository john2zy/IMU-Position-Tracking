import matplotlib.pyplot as plt

def plot_signal(a, a_filtered, w, w_filtered, m):
    '''
    a dirty warpper to make main notebook cleaner
    '''
    
    # plot acc and angular velocity
    fig, ax = plt.subplots(nrows=3, ncols=3)
    ax[0, 0].plot(a[:, 0], label='$a_x$')
    ax[0, 0].plot(a_filtered[: ,0], label='Filtered $a_x$')
    ax[1, 0].plot(a[:, 1], label='$a_y$')
    ax[1, 0].plot(a_filtered[: ,1], label='Filtered $a_y$')
    ax[2, 0].plot(a[:, 2], label='$a_z$')
    ax[2, 0].plot(a_filtered[: ,2], label='Filtered $a_z$')

    ax[0, 1].plot(w[:, 0], label='$w_x$')
    ax[0, 1].plot(w_filtered[: ,0], label='Filtered $w_x$')
    ax[1, 1].plot(w[:, 1], label='$w_y$')
    ax[1, 1].plot(w_filtered[: ,1], label='Filtered $w_x$')
    ax[2, 1].plot(w[:, 2], label='$w_z$')
    ax[2, 1].plot(w_filtered[: ,2], label='Filtered $w_x$')

    ax[0, 2].plot(m[:, 0], label='$m_x$')
    ax[1, 2].plot(m[:, 1], label='$m_y$')
    ax[2, 2].plot(m[:, 2], label='$m_z$')

    # for i in range(3):
    #     for j in range(3):
    #         ax[i, j].legend(loc='upper left')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
    
    
def plot3D_g_and_mag(gn, mn):
    '''
    plot initial gravity and magnectic field direction
    '''
    
    g_fig = plt.figure()
    ax = g_fig.add_subplot(111, projection='3d')

    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_zlim(-12, 12)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.plot(gn[0], gn[1], gn[2], 'o')
    ax.plot(mn[0], mn[1], mn[2], 'o')
    ax.plot([0], [0], [0], 'ro')

    plt.show()
    
    
def plot_g_and_acc(g, ab):
    '''
    plot tracked gravity and body frame acceleration
    '''
    
    fig, ax = plt.subplots(nrows=1, ncols=3)

    ax[0].plot(g[:, 0], 'r-', label='$g_x$')
    ax[0].plot(ab[:, 0], label="$a^b_x$")
    ax[0].legend(loc='upper left')
    ax[0].set_ylim(-12, 12)

    ax[1].plot(g[:, 1], 'g-', label='$g_y$')
    ax[1].plot(ab[:, 1], label="$a^b_y$")
    ax[1].legend(loc='upper left')
    ax[1].set_ylim(-12, 12)

    ax[2].plot(g[:, 2], 'b-', label='$g_z$')
    ax[2].plot(ab[:, 2], label="$a^b_z$")
    ax[2].legend(loc='upper left')
    ax[2].set_ylim(-12, 12)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
    
    
def plot_3D(data, lim: dict = {}):
    '''
    @param data: [[data, label string], ...]
    @param lim: {'x': [xliml, xlimr], ...}
    '''
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for item in data:
        name = item[1]
        d = item[0]
        ax.plot(d[:, 0], d[:, 1], d[:, 2], 'o', label=name)
    
    if lim.get('x') is not None:
        ax.set_xlim(lim['x'][0], lim['x'][1])
    if lim.get('y') is not None:
        ax.set_xlim(lim['y'][0], lim['x'][1])
    if lim.get('z') is not None:
        ax.set_xlim(lim['z'][0], lim['x'][1])
        
    ax.legend()
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.plot([0], [0], [0], 'ro')