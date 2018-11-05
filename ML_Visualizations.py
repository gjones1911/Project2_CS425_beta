import matplotlib
import matplotlib.pyplot as plt
import numpy


def make_scree_graph_data(np_data_array):
    u, s, vh = numpy.linalg.svd(np_data_array, full_matrices=True, compute_uv=True)
    v = numpy.transpose(vh)

    sum_s = sum(s.tolist())
    s_sum = numpy.cumsum(s)[-1]
    print('shape of s')
    print(s.shape)
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


def make_scree_plot_usv(s, num_obs):
    sum_s = sum(s.tolist())
    s_sum = numpy.cumsum(s)[-1]
    #print('shape of s')
    #print('There are {:d} observations and {:d} variables or attributes'.format(num_obs, num_var))

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


def make_prop_o_var_plot(s, num_obs):

    sum_s = sum(s.tolist())

    ss = s**2

    sum_ss = sum(ss.tolist())

    prop_list = list()

    for i in range(1, num_obs+1):
        #print(ss[0:i].tolist())
        #print('')
        prop_list.append(sum(s[0:i].tolist()) / sum_s)

        #print('Prop. of Var. with k = {:d}'.format(i+1))
        #print(prop_list[-1])

    print('length of prop list: {:d}'.format(len(prop_list)))

    single_vals = numpy.arange(num_obs)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(single_vals, prop_list, 'ro-', linewidth=2)
    plt.title('Proportion of Variance')
    plt.xlabel('Eigenvectors')
    plt.ylabel('Prop. of var.')
    leg = plt.legend(['Eigenvectors vs. Prop. of Var.'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    plt.show()

    return


def dual_scree_prop_var(s, num_obs):
    sum_s = sum(s.tolist())

    eigen_vals = s ** 2 /sum_s

    single_vals = numpy.arange(num_obs)

    prop_list = list()

    for i in range(1, num_obs + 1):
        prop_list.append(sum(s[0:i].tolist()) / sum_s)


    fig, axs = plt.subplots(2,1)
    #plt.figure(1)
    #fig = plt.figure(figsize=(8, 5))
    axs[0].plot(single_vals, eigen_vals, 'ro-',  linewidth=2)
    plt.title('Scree Plot')
    #axs[0].title('Scree Plot')
    axs[0].set_xlabel('Principal Component')
    axs[0].set_ylabel('Eigenvalue')
    leg = axs[0].legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)

    #plt.figure(2)
    axs[1].plot(single_vals, prop_list, 'go-', linewidth=2)
    #plt.title('Proportion of Variance')
    axs[1].set_xlabel('Eigenvectors')
    axs[1].set_ylabel('Prop. of var.')
    leg = plt.legend(['Eigenvectors vs. Prop. of Var.'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    '''
    plt.subplot(2,2,1)
    #fig = plt.figure(figsize=(8, 5))
    plt.plot(single_vals, prop_list, 'ro-', linewidth=2)
    plt.title('Proportion of Variance')
    plt.xlabel('Eigenvectors')
    plt.ylabel('Prop. of var.')
    leg = plt.legend(['Eigenvectors vs. Prop. of Var.'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    '''

    plt.show()


    plt.show()

    return


def basic_scatter_plot(x, y, x_label, y_label, title, legend):

    fig = plt.figure(figsize=(8, 5))
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    leg = plt.legend([legend], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    plt.show()
    return
