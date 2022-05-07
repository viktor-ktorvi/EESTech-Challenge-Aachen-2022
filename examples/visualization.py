from matplotlib import pyplot as plt

from utils import load_radar_dict


def plot_recording(radar_files_dict, fig, ax, color=None):
    assert len(ax) == 4
    for i in range(3):
        ax[i].plot(radar_files_dict['abses'][i, :], color=color)
        ax[i].set_title('Abses {:d}'.format(i))

    ax[3].plot(radar_files_dict['phases'], color=color)
    ax[3].set_title('Phases')

    fig.tight_layout()

    return fig


def visualize(folder, filenames, title, colors=None, figsize=None):
    """
    Visualize one or multiple recordings. Plots abses[0..2] and phase
    :param folder: str
    :param filenames: str or list of str
    :param title: str
    :param colors: str or list of str
    :param figsize: tuple
    :return: figure
    """
    if type(filenames) is not list:
        filenames = [filenames]

        assert type(colors) is not list
        colors = [colors]
    else:
        if colors is None:
            colors = [None] * len(filenames)

    if figsize is not None:
        fig, ax = plt.subplots(4, 1, figsize=figsize)
    else:
        fig, ax = plt.subplots(4, 1)
    fig.suptitle(title)
    for i in range(len(filenames)):
        radar_files_dict = load_radar_dict(folder, filenames[i])
        fig = plot_recording(radar_files_dict, fig, ax, color=colors[i])

    return fig


# %%
if __name__ == '__main__':
    figsize = (6, 6)
    folder = 'recordings'
    filename = 'farness_0_moving_1_radar_data_2022_05_06_20_24_36.npz'

    title = 'Visualize one'
    color = 'blue'

    fig = visualize(folder, filename, title, color, figsize=figsize)
    fig.show()

    # %%
    folder = 'recordings'
    filenames = ['farness_1_moving_1_radar_data_2022_05_06_20_25_06.npz',
                 'farness_0_moving_1_radar_data_2022_05_06_20_24_36.npz']

    colors = ['b', 'r']
    title = 'Compare'
    fig = visualize(folder, filenames, title, colors, figsize=figsize)
    fig.show()
