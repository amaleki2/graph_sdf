import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data_utils import get_rasterized_points, get_sdf, preprocess_mesh
from src.train_utils import get_device
from src.render.diff_sdf_render import render_surface_img
from src.test_utils import sdf_grid_to_surface_mesh


def generate_target_sdf_from_mesh(mesh_file, res, n_jobs=4, n_splits=10, save_name=None):
    import trimesh
    from joblib import delayed, Parallel

    mesh = trimesh.load(mesh_file)
    mesh = preprocess_mesh(mesh, mesh_out=None, merge_vertex=False, with_scaling_to_unit_box=True, scaler=1.8)
    print(mesh.vertices.min(axis=0), mesh.vertices.max(axis=0))

    def func(x):
        return get_sdf(mesh, x)

    points = get_rasterized_points(res)
    points_split = np.array_split(points, n_splits)

    sdf_values = Parallel(n_jobs=n_jobs)(delayed(func)(x) for x in tqdm.tqdm(points_split))
    sdf_values = np.concatenate(sdf_values).reshape(res, res, res)
    if save_name is None:
        sdf_grid_to_surface_mesh(sdf_values, save_name=None, level=0)
    else:
        np.save(save_name, sdf_values)
    return sdf_values


def plot_results(img_pred, img_truth, save_fig=None):
    img_pred_np = img_pred.detach().cpu().numpy()
    img_truth_np = img_truth.detach().cpu().numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(img_pred_np)
    plt.title('prediction')

    plt.subplot(1, 2, 2)
    plt.imshow(img_truth_np)
    plt.title('ground truth')

    if save_fig is None:
        plt.show()
    else:
        plt.savefig(save_fig)


res = 64
lr = 0.001
n_itr_max = 2000
plot_every = 50
sdf_target_name = "test/objects/dolphin%d.npy" % res
device = get_device('cuda')[0]

sdf_target = np.load(sdf_target_name)
sdf_target = torch.from_numpy(sdf_target).to(device=device, dtype=torch.float32)

xyz = get_rasterized_points(res)
sdf = np.linalg.norm(xyz, axis=1) - 0.5
sdf = sdf.reshape(res, res, res)
sdf_pred = torch.from_numpy(sdf).to(device=device, dtype=torch.float32).requires_grad_(True)

optimizer = torch.optim.Adam([sdf_pred], lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.2)

mse_loss_func = torch.nn.MSELoss()

camera_pos = [1.2, 1.2, 1.2]

for itr in range(n_itr_max):
    optimizer.zero_grad()
    render_pred = render_surface_img(sdf_pred, camera_pos=camera_pos, box=None, img_size=None)
    render_target = render_surface_img(sdf_target, camera_pos=camera_pos, box=None, img_size=None)

    loss = mse_loss_func(render_pred, render_target)
    loss.backward()
    optimizer.step()
    scheduler.step()

    sdf_loss = mse_loss_func(sdf_pred, sdf_target)

    if itr % plot_every == 0:
        print("itr=%d, lr=%0.6e render loss=%0.4f, sdf loss=%0.4f" %
              (itr, optimizer.param_groups[0]['lr'], loss.item(), sdf_loss.item())
              )
        plot_results(render_pred, render_target)






