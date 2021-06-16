from torch_geometric.nn import DataParallel
from .train_utils import *


def train_sdf(model,
              train_dl,
              test_dl,
              eikonal_coef=1,
              loss_func='l1',
              n_epochs=500,
              print_every=25,
              save_every=25,
              device='cpu',
              save_folder_name="save",
              optimizer='Adam',
              lr_0=0.001,
              lr_scheduler_params=None):

    lr_scheduler_params = {} if lr_scheduler_params is None else lr_scheduler_params

    device, data_parallel = get_device(device)
    optimizer             = get_optimizer(model, optimizer, lr_0)
    scheduler             = get_scheduler(optimizer, **lr_scheduler_params)
    loss_func             = get_loss_func(loss_func, data_parallel)
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

            data.x.requires_grad = True
            data.x.retain_grad()

            pred = model(data)
            loss = loss_func(pred)
            loss += eikonal_loss_func(data, pred) * eikonal_coef
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_epoch_loss = sum(epoch_loss) / len(train_dl)
        tf_writer.add_scalar("Loss/train", train_epoch_loss, epoch)

        test_epoch_loss_mean = None
        if epoch % print_every == 0 and len(test_dl) > 0:
            test_epoch_loss = []
            with torch.no_grad():
                for data in test_dl:
                    model.eval()
                    if not data_parallel:
                        data = data.to(device)
                    pred = model(data)
                    loss = loss_func(pred)
                    test_epoch_loss.append(loss.item())

            test_epoch_loss_mean = np.mean(test_epoch_loss)
            tf_writer.add_scalar("Loss/test", test_epoch_loss_mean, epoch)

        print_to_screen(epoch, optimizer, train_epoch_loss, test_loss=test_epoch_loss_mean)

        if epoch % save_every == 0:
            save_latest(model, epoch, optimizer, save_folder_name, data_parallel)

        scheduler.step()
