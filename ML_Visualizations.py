import matplotlib
import matplotlib.pyplot as plt
import numpy


def make_scree_graph_data(np_data_array):
    u, s, vh = numpy.linalg.svd(np_data_array, full_matrices=True, compute_uv=True)
    v = numpy.transpose(vh)

    sum_s = sum(s.tolist())
    s_sum = numpy.cumsum(s)[-1]
    print('shape of s')
    obs_var = np_data_array.shape
    num_obs = obs_var[0]
    num_var = obs_var[1]

    print('There are {:d} observations and {:d} variables or attributes'.format(num_obs, num_var))

    eigen_vals = s ** 2 /s_sum

    single_vals = numpy.arange(num_obs)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(single_vals, eigen_vals, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    # I don't like the default legend so I typically make mine like below, e.g.
    # with smaller fonts and a bit transparent so I do not cover up data, and make
    # it moveable by the viewer in case upper-right is a bad place for it
    leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    plt.show()
    return u, s, vh, v


def make_scree_plot_usv(s, num_obs, num_var):
    sum_s = sum(s.tolist())
    s_sum = numpy.cumsum(s)[-1]
    print('shape of s')
    print('There are {:d} observations and {:d} variables or attributes'.format(num_obs, num_var))

    eigen_vals = s ** 2 / s_sum

    single_vals = numpy.arange(num_obs)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(single_vals, eigen_vals, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    plt.show()
    return


def make_prop_o_var_plot(s):
    i = 0
    return