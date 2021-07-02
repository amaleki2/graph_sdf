from torch_geometric.nn import DataParallel
from .train_utils import *


def train_sdf(model,
              train_dl,
              test_dl,
              pretrained_model_weights=None,
              plot_gradients=False,
              loss_funcs=None,
              clip_gradients=None,
              n_epochs=500,
              print_every=25,
              save_every=25,
              device='cpu',
              save_folder_name='save',
              optimizer='Adam',
              lr_0=0.001,
              lr_scheduler_params=None):

    if lr_scheduler_params is None:
        lr_scheduler_params = {}

    device, data_parallel = get_device(device)
    optimizer = get_optimizer(model, optimizer, lr_0)
    scheduler = get_scheduler(optimizer, **lr_scheduler_params)
    composite_loss_func = get_loss_funcs(loss_funcs, data_parallel)
    clip_gradients_func = get_clip_gradients_func(clip_gradients)
    tf_writer = get_summary_writer(save_folder_name)

    if pretrained_model_weights:
        model_path = os.path.join(os.getcwd(), save_folder_name, pretrained_model_weights)
        model_weights = torch.load(model_path, map_location=device)["model_state_dict"]
        model.load_state_dict(model_weights)

    if data_parallel:
        device0 = torch.device('cuda:%d'%device[0])
        model = DataParallel(model, device_ids=device)
        model = model.to(device0)
    else:
        model = model.to(device=device)

    for epoch in range(n_epochs + 1):
        train_epoch_losses = []
        for data in train_dl:
            model.train()
            optimizer.zero_grad()
            if not data_parallel:
                data = data.to(device)
            pred = model(data)
            losses = composite_loss_func(pred, model, data)
            loss = sum(losses.values())
            train_epoch_losses.append(losses)
            loss.backward()
            if clip_gradients is not None:
                clip_gradients_func(model.parameters())
            optimizer.step()

        if plot_gradients:
            write_gradients_to_file(model.named_parameters(), epoch, save_folder_name)

        write_to_tensorboard(epoch, train_epoch_losses, tf_writer, 'train')

        test_epoch_losses = []
        if epoch % print_every == 0 and len(test_dl) > 0:
            with torch.no_grad():
                for data in test_dl:
                    model.eval()
                    if not data_parallel:
                        data = data.to(device)
                    pred = model(data)

                    test_losses = composite_loss_func(pred, model, data)
                    test_epoch_losses.append(test_losses)

            write_to_tensorboard(epoch, test_epoch_losses, tf_writer, 'test')

        write_to_screen(epoch, optimizer, train_epoch_losses, test_losses=test_epoch_losses)

        if epoch % save_every == 0:
            save_latest(model, epoch, optimizer, save_folder_name, data_parallel)

        scheduler.step()
