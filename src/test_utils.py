import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tensorflow.python.summary.summary_iterator import summary_iterator


def load_latest(model, save_dir, device):
    model_latest_path = os.path.join(os.getcwd(), save_dir, "model_latest.pth")
    model_weights = torch.load(model_latest_path, map_location=device)["model_state_dict"]
    model.load_state_dict(model_weights)
    return model


def parse_event_file(event_file):
    train = []
    test = []
    wall_time_begin = np.inf
    wall_time_end = -np.inf
    for summary in summary_iterator(event_file):
        step = summary.step
        wall_time = summary.wall_time
        wall_time_begin = min(wall_time_begin, wall_time)
        wall_time_end = max(wall_time_end, wall_time)
        if len(summary.summary.value) > 0:
            value = summary.summary.value[0]
            if hasattr(value, 'tag'):
                if value.tag == "Loss/train":
                    train_loss = value.simple_value
                    train.append([step, train_loss])
                elif value.tag == "Loss/test":
                    test_loss = value.simple_value
                    test.append([step, test_loss])
    train, test, train_time = np.array(train), np.array(test), (wall_time_end - wall_time_begin) / 3600

    return train, test, train_time


def load_events(save_dir):
    save_folder = os.path.join(os.getcwd(), save_dir)
    event_file = [os.path.join(save_folder, x) for x in os.listdir(save_folder) if x.startswith('events')]
    if len(event_file) == 0:
        print("no event file was found")
        return

    if len(event_file) > 1:
        print("more than one event file was found. used the latest")
    event_file = event_file[-1]
    return event_file


def plot_losses_from_event_file(save_dir, save_files=False):
    event_file = load_events(save_dir)
    train_hist, test_hist, train_time = parse_event_file(event_file)
    plt.figure(figsize=(6, 6))
    plt.semilogy(train_hist[:, 0], train_hist[:, 1], 'k')
    plt.semilogy(test_hist[:, 0], test_hist[:, 1], 'k--')
    plt.title('training time was %0.2f hours' % train_time)
    if save_files:
        save_name = os.path.join(os.getcwd(), save_dir, 'loss.png')
        plt.savefig(save_name)
    else:
        plt.show()


def plot_scatter_contour_3d(points, true_vals, pred_vals, levels=None, save_name=None):
    ends, n_pts = 0.9, 100
    n_pnts_c = n_pts * 1j
    x = np.linspace(-ends, ends, n_pts, endpoint=True)
    X, Y, Z = np.mgrid[-ends:ends:n_pnts_c, -ends:ends:n_pnts_c, -ends:ends:n_pnts_c]
    SDFS_true = griddata(points, true_vals, (X, Y, Z))
    SDFS_pred = griddata(points, pred_vals, (X, Y, Z))
    fig, axes = plt.subplots(figsize=(25, 15), nrows=3, ncols=3)
    for i in range(3):
        ax1, ax2, ax3 = axes[i]
        z_slice = 40 * i + 10
        cntr1 = ax1.contour(x, x, SDFS_true[:, :, z_slice], levels=levels, linewidths=1, colors='k')
        plt.clabel(cntr1, fmt='%0.2f', colors='k', fontsize=10)
        cntr1 = ax1.contourf(x, x, SDFS_true[:, :, z_slice], cmap="RdBu_r", levels=20)
        fig.colorbar(cntr1, ax=ax1)
        ax1.set(xlim=(-1, 1), ylim=(-1, 1))
        ax1.set_xticks([])
        ax1.set_yticks([])

        cntr2 = ax2.contour(x, x, SDFS_pred[:, :, z_slice], levels=levels, linewidths=1, colors='k')
        plt.clabel(cntr2, fmt='%0.2f', colors='k', fontsize=10)
        cntr2 = ax2.contourf(x, x, SDFS_pred[:, :, z_slice], cmap="RdBu_r", levels=20)
        fig.colorbar(cntr2, ax=ax2)
        ax2.set(xlim=(-1, 1), ylim=(-1, 1))
        ax2.set_xticks([])
        ax2.set_yticks([])
        #     if levels:
        #         new_levels = [(l + r) / 2 for (l, r) in zip(levels[1:], levels[:-1])] + levels
        #         new_levels = sorted(new_levels)
        #     else:
        #         new_levels = None
        new_levels = levels
        cntr3 = ax3.contour(x, x, SDFS_true[:, :, z_slice], levels=new_levels, linewidths=2, colors='k')
        plt.clabel(cntr3, fmt='%0.2f', colors='k', fontsize=10)
        cntr3 = ax3.contour(x, x, SDFS_pred[:, :, z_slice], levels=new_levels, linewidths=1, colors='r', linestyles='--')
        ax3.set(xlim=(-1, 1), ylim=(-1, 1))
        ax3.set_xticks([])
        ax3.set_yticks([])
    plt.subplots_adjust(wspace=0.25)

    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name)