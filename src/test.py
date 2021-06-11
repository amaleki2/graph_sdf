from .train_utils import get_device
from .test_utils import *


def test_and_visualize(model,
                       data_dl,
                       device='cpu',
                       save_folder_name="save",
                       save_files=False):

    plot_losses_from_event_file(save_folder_name, save_files=save_files)

    device, _ = get_device(device)
    model = model.to(device=device)
    model = load_latest(model, save_folder_name, device)

    predictions = []
    with torch.no_grad():
        for i, data in enumerate(data_dl):
            model.eval()
            data = data.to(device)
            points = data.x[:, :3].cpu().numpy()
            pred_val = model(data).x.cpu().numpy().reshape(-1)
            true_val = data.y.cpu().numpy().reshape(-1)
            if save_files:
                save_name = os.path.join(os.getcwd(), save_folder_name, 'fig%d.png'%i)
            else:
                save_name = None
            plot_scatter_contour_3d(points, true_val, pred_val, save_name=save_name)
            predictions.append([points, true_val, pred_val])