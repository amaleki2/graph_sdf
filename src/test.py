from tqdm import tqdm
from .train_utils import get_device
from .test_utils import *


def test_and_visualize(model,
                       data_dl,
                       device='cuda',
                       save_losses=True,
                       save_3d_surface=True,
                       save_2d_contours=True,
                       save_folder_name="save"):

    if save_losses:
        plot_losses(save_folder_name)

    device, _ = get_device(device)
    model = model.to(device=device)
    model = load_latest(model, save_folder_name, device)

    predictions = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_dl)):
            model.eval()
            data = data.to(device)
            points = data.x[:, :3].cpu().numpy()
            pred_val, pred_sgn = model(data).x.cpu().numpy().T
            true_val = data.y.cpu().numpy().reshape(-1)
            true_sgn = true_val >= 0

            if save_2d_contours:
                interpolate = not save_3d_surface  # if true, already grid format, no interpolation necessary.
                save_name_fig = os.path.join(os.getcwd(), save_folder_name, 'fig%d.jpg'%i)
                plot_2d_contours(points, true_val, pred_val, save_name=save_name_fig,
                                 levels=np.linspace(-0.45, 0.45, 7), interpolate=interpolate)

                save_name_fig = os.path.join(os.getcwd(), save_folder_name, 'fig_sgn%d.jpg' % i)
                mask = true_val != 0
                plot_2d_contours_sgn(points, true_sgn, pred_sgn, mask=mask,
                                     save_name=save_name_fig, interpolate=interpolate)

            if save_3d_surface:
                save_name_msh = [os.path.join(os.getcwd(), save_folder_name, 'true_surface_mesh%d.obj'%i),
                                 os.path.join(os.getcwd(), save_folder_name, 'pred_surface_mesh%d.obj' % i)]
                plot_surface_mesh(true_val, pred_val, save_names=save_name_msh, level=0)


            predictions.append([points, true_val, pred_val])

    return predictions
