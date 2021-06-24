import os
import glob
import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from skimage.measure import marching_cubes
from tensorflow.python.summary.summary_iterator import summary_iterator


def load_latest(model, save_dir, device):
    model_latest_path = os.path.join(os.getcwd(), save_dir, "model_latest.pth")
    model_weights = torch.load(model_latest_path, map_location=device)["model_state_dict"]
    model.load_state_dict(model_weights)
    return model


def parse_event_file(event_file):
    train = []
    test = []
    tag = os.path.split(os.path.split(event_file)[0])[1]
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
                if value.tag == "train":
                    train_loss = value.simple_value
                    train.append([step, train_loss])
                elif value.tag == "test":
                    test_loss = value.simple_value
                    test.append([step, test_loss])
    train, test, train_time = np.array(train), np.array(test), (wall_time_end - wall_time_begin) / 3600
    return train, test, train_time, tag


def get_all_event_files(save_dir):
    templates = [save_dir + "*events.out.tfevents.*",
                 save_dir + "*/*events.out.tfevents.*",
                 save_dir + "*/*/*events.out.tfevents.*"]
    all_event_files = []
    for template in templates:
        all_event_files += list(glob.iglob(template))
    all_event_files = [os.path.join(os.getcwd(), x) for x in all_event_files]
    return all_event_files


def plot_and_save_events(event, save_dir, save_files=True):
    train_hist, test_hist, train_time, tag = event
    if len(train_hist) > 0:
        data = train_hist
    elif len(test_hist) > 0:
        data = test_hist
    else:
        return

    plt.figure(figsize=(8, 6))
    plt.semilogy(data[:, 0], data[:, 1], 'k')
    plt.xlabel("epochs")
    plt.ylabel(tag)
    plt.title("training time was %0.2f hours" % train_time)

    if save_files:
        save_name = os.path.join(os.getcwd(), save_dir, tag + ".jpg")
        plt.savefig(save_name)
    else:
        plt.show()


def plot_losses(save_dir):
    all_event_files = get_all_event_files(save_dir)
    for event_file in all_event_files:
        event = parse_event_file(event_file)
        plot_and_save_events(event, save_dir)


def plot_2d_contours(points, true_vals, pred_vals, levels=None, save_name=None, interpolate=True):
    if interpolate:
        ends, grid_size = 0.9, 100
        n_pnts_c = grid_size * 1j
        x = np.linspace(-ends, ends, grid_size, endpoint=True)
        X, Y, Z = np.mgrid[-ends:ends:n_pnts_c, -ends:ends:n_pnts_c, -ends:ends:n_pnts_c]
        SDFS_true = griddata(points, true_vals, (X, Y, Z))
        SDFS_pred = griddata(points, pred_vals, (X, Y, Z))
    else:  # already in grid format
        volume_points_mask = true_vals != 0
        n_volume_points = volume_points_mask.sum()
        grid_size = round(n_volume_points ** (1 / 3))
        assert grid_size ** 3 == n_volume_points

        x = np.linspace(-1, 1, grid_size, endpoint=True)
        SDFS_true = true_vals[volume_points_mask].reshape(grid_size, grid_size, grid_size)
        SDFS_pred = pred_vals[volume_points_mask].reshape(grid_size, grid_size, grid_size)

    fig, axes = plt.subplots(figsize=(25, 15), nrows=3, ncols=3)
    for i in range(3):
        ax1, ax2, ax3 = axes[i]
        z_slice = round((0.1 + 0.4*i) * grid_size)
        cntr1 = ax1.contour(x, x, SDFS_true[:, :, z_slice], levels=levels, linewidths=1, colors='k')
        plt.clabel(cntr1, fmt='%0.2f', colors='k', fontsize=10)
        cntr1 = ax1.contourf(x, x, SDFS_true[:, :, z_slice], cmap="RdBu_r", levels=20)
        fig.colorbar(cntr1, ax=ax1)
        ax1.set(xlim=(-1, 1), ylim=(-1, 1))
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.title.set_text('Ground Truth at z=%0.2f' %x[z_slice])

        cntr2 = ax2.contour(x, x, SDFS_pred[:, :, z_slice], levels=levels, linewidths=1, colors='k')
        plt.clabel(cntr2, fmt='%0.2f', colors='k', fontsize=10)
        cntr2 = ax2.contourf(x, x, SDFS_pred[:, :, z_slice], cmap="RdBu_r", levels=20)
        fig.colorbar(cntr2, ax=ax2)
        ax2.set(xlim=(-1, 1), ylim=(-1, 1))
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.title.set_text('Prediction at z=%0.2f' %x[z_slice])
        #     if levels:
        #         new_levels = [(l + r) / 2 for (l, r) in zip(levels[1:], levels[:-1])] + levels
        #         new_levels = sorted(new_levels)
        #     else:
        #         new_levels = None
        new_levels = levels
        cntr3 = ax3.contour(x, x, SDFS_true[:, :, z_slice], levels=new_levels, linewidths=2, colors='k')
        plt.clabel(cntr3, fmt='%0.2f', colors='k', fontsize=10)
        cntr3 = ax3.contour(x, x, SDFS_pred[:, :, z_slice], levels=new_levels, linewidths=1, colors='r')
        ax3.set(xlim=(-1, 1), ylim=(-1, 1))
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.title.set_text('Comparison at z=%0.2f' %x[z_slice])
    plt.subplots_adjust(wspace=0.25)
    fig.suptitle('Results with %d volume points' % (true_vals != 0).sum())

    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name)


def sdf_grid_to_surface_mesh(grid_sdf, save_name=None, level=1):
    verts, faces, normals, values = marching_cubes(grid_sdf, level=level)
    mesh = trimesh.Trimesh(verts, faces)
    if save_name is None:
        mesh.show()
    else:
        with open(save_name, 'wb') as fid:
            mesh.export(fid, file_type='obj')


def plot_surface_mesh(true_sdfs, pred_sdfs, save_names=None, level=1):
    volume_points_mask = true_sdfs != 0
    n_volume_points = volume_points_mask.sum()
    grid_size = round(n_volume_points ** (1/3))
    assert grid_size ** 3 == n_volume_points

    grid_true_sdfs = true_sdfs[volume_points_mask].reshape(grid_size, grid_size, grid_size)
    sdf_grid_to_surface_mesh(grid_true_sdfs, save_name=save_names[0], level=level)

    grid_pred_sdfs = pred_sdfs[volume_points_mask].reshape(grid_size, grid_size, grid_size)
    sdf_grid_to_surface_mesh(grid_pred_sdfs, save_name=save_names[1], level=level)


