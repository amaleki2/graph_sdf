from torch_geometric.nn import DataParallel
from .train_utils import *


def train_sdf(model,
              train_dl,
              test_dl,
              loss_funcs='l1',
              loss_funcs_coefs=None,
              n_epochs=500,
              print_every=25,
              save_every=25,
              device='cpu',
              save_folder_name="save",
              optimizer='Adam',
              lr_0=0.001,
              lr_scheduler_params=None):

    if not isinstance(loss_funcs, list):
        loss_funcs = [loss_funcs]

    if lr_scheduler_params is None:
        lr_scheduler_params = {}

    if loss_funcs_coefs is None:
        loss_funcs_coefs = [1.0] * len(loss_funcs)

    device, data_parallel = get_device(device)
    optimizer             = get_optimizer(model, optimizer, lr_0)
    scheduler             = get_scheduler(optimizer, **lr_scheduler_params)
    loss_funcs            = get_loss_funcs(loss_funcs, data_parallel)
    tf_writer             = get_summary_writer(save_folder_name)

    if data_parallel:
        device0 = torch.device('cuda:%d'%device[0])
        model = DataParallel(model, device_ids=device)
        model = model.to(device0)
    else:
        model = model.to(device=device)

    for epoch in range(n_epochs + 1):
        epoch_loss = []
        for data in train_dl:
            model.train()
            optimizer.zero_grad()
            if not data_parallel:
                data = data.to(device)
            pred = model(data)
            losses = [func(pred) for func in loss_funcs]
            epoch_loss.append(losses)
            loss = sum([c * l for (c, l) in zip(loss_funcs_coefs, losses)])
            loss.backward()
            optimizer.step()

        train_epoch_loss_mean = torch.tensor(epoch_loss).mean(dim=0)
        for i_loss, loss_mean in enumerate(train_epoch_loss_mean):
           tf_writer.add_scalar("Loss_%d/train"%i_loss, loss_mean.item(), epoch)

        test_epoch_loss_mean = None
        if epoch % print_every == 0 and len(test_dl) > 0:
            test_epoch_loss = []
            with torch.no_grad():
                for data in test_dl:
                    model.eval()
                    if not data_parallel:
                        data = data.to(device)
                    pred = model(data)
                    test_losses = [func(pred) for func in loss_funcs]
                    test_epoch_loss.append(test_losses)

            test_epoch_loss_mean = torch.tensor(test_epoch_loss).mean(dim=0)
            for i_loss, loss_mean in enumerate(test_epoch_loss_mean):
                tf_writer.add_scalar("Loss_%d/test" % i_loss, loss_mean.item(), epoch)

        print_to_screen(epoch, optimizer, train_epoch_loss_mean, test_loss=test_epoch_loss_mean)

        if epoch % save_every == 0:
            save_latest(model, epoch, optimizer, save_folder_name, data_parallel)

        scheduler.step()
