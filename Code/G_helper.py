def value_mean(X, y, c, t):
    indices = np.logical_and(X[:, 0] == c, X[:, 1] == t)

    if sum(indices) == 0:
        return -1
    else:
        return np.nanmean(y[indices])


def plot_bar(tab_means, X, y):
    # surface plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    y_axis = np.arange(tab_means.shape[0])
    x_axis = np.arange(tab_means.shape[1])
    X_axis, Y_axis = np.meshgrid(x_axis, y_axis)

    Z_axis = np.zeros_like(X_axis)
    for c in y_axis:
        for t in x_axis:
            Z_axis[c, t] = value_mean(X, y, c, t)

    indices = Z_axis > -1

    z_pos = [0] * sum(sum(indices))
    x_size = [1] * sum(sum(indices))
    y_size = [1] * sum(sum(indices))

    ax.bar3d(X_axis[indices], Y_axis[indices], z_pos, x_size, y_size, Z_axis[indices])

    ax.set_title('Mean of Value Function')
    ax.set_xlabel('Time')
    ax.set_ylabel('Capacity')
    plt.yticks(np.arange(0, c_max + 1, 1.0))

    plt.show()


def plot_surface(tab_means, X, y):
    # surface plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    y_axis = np.arange(tab_means.shape[0])
    x_axis = np.arange(tab_means.shape[1])
    X_axis, Y_axis = np.meshgrid(x_axis, y_axis)

    Z_axis = np.zeros_like(X_axis)
    for c in y_axis:
        for t in x_axis:
            Z_axis[c, t] = value_mean(X, y, c, t)

    ax.plot_surface(X_axis, Y_axis, Z_axis, cmap='viridis')

    ax.set_title('Mean of Value Function')
    ax.set_xlabel('Time')
    ax.set_ylabel('Capacity')
    plt.yticks(np.arange(0, c_max + 1, 1.0))

    plt.show()


def value_analysis(c_samples, X, y):
    c_max = int(np.max(c_samples))

    tab_means = np.zeros((c_max + 1, T))
    tab_numbers = np.zeros((c_max + 1, T))

    for c in np.arange(tab_means.shape[0]):
        for t in np.arange(tab_means.shape[1]):
            indices = np.logical_and(X[:, 0] == c, X[:, 1] == t)
            tab_means[c_max - c, t] = value_mean(X, y, c, t)
            tab_numbers[c_max - c, t] = sum(indices)

    df_means = pd.DataFrame(tab_means,
                            columns=["t" + str(t) for t in np.arange(tab_means.shape[1])],
                            index=["c" + str(c) for c in (c_max - np.arange(tab_means.shape[0]))])
    df_numbers = pd.DataFrame(tab_numbers,
                              columns=["t" + str(t) for t in np.arange(tab_numbers.shape[1])],
                              index=["c" + str(c) for c in (c_max - np.arange(tab_numbers.shape[0]))])

    return df_means, df_numbers, tab_means


