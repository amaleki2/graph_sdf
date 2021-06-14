from tqdm import tqdm
from .train_utils import get_device
from .test_utils import *


def test_and_visualize(model,
                       data_dl,
                       device='cuda',
                       save_folder_name="save",
                       save_files=False,
                       with_3d_surface=False):

    plot_losses_from_event_file(save_folder_name, save_files=save_files)

    device, _ = get_device(device)
    model = model.to(device=device)
    model = load_latest(model, save_folder_name, device)

    predictions = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_dl)):
            model.eval()
            data = data.to(device)
            points = data.x[:, :3].cpu().numpy()
            pred_val = model(data).x.cpu().numpy().reshape(-1)
            true_val = data.y.cpu().numpy().reshape(-1)

            if save_files:
                save_name_fig = os.path.join(os.getcwd(), save_folder_name, 'fig%d.png'%i)
                save_name_msh = [os.path.join(os.getcwd(), save_folder_name, 'true_surface_mesh%d.obj'%i),
                                 os.path.join(os.getcwd(), save_folder_name, 'pred_surface_mesh%d.obj' % i)]
            else:
                save_name_fig = None
                save_name_msh = None

            if with_3d_surface:
                plot_surface_mesh(true_val, pred_val, save_names=save_name_msh, level=0)
            else:
                plot_scatter_contour_3d(points, true_val, pred_val, save_name=save_name_fig,
                                        levels=np.linspace(-0.45, 0.45, 7))

            predictions.append([points, true_val, pred_val])

    return predictions
