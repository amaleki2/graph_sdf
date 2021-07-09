import os
import sys
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.tensorboard import SummaryWriter
from src.data_utils import compute_edge_features


def find_best_gpu():
    # this function finds the GPU with most free memory.
    if 'linux' in sys.platform and torch.cuda.device_count() > 1:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        os.remove('tmp')
        gpu_id = np.argmax(memory_available).item()
        print("best gpu is %d with %0.1f Gb available space" %(gpu_id, memory_available[gpu_id] / 1000))
        return gpu_id


def get_device(device):
    if device == 'cpu':
        print('using cpu')
        device = torch.device('cpu')
    elif device == 'cuda':
        print('using gpu: ', end="")
        device = torch.device('cuda')
        gpu_id = find_best_gpu()
        if gpu_id:
            torch.cuda.set_device(gpu_id)
    else:
        try:
            print('training using multiple gpus: ' + device)
            device = [int(x) for x in device.split()]
        except:
            raise ValueError("device is not correct.")

    data_parallel = isinstance(device, list)
    return device, data_parallel


def get_loss_func_aggr(loss_func_aggr):
    if loss_func_aggr == 'l1':
        loss_func = torch.nn.L1Loss()
    elif loss_func_aggr == 'l2':
        loss_func = torch.nn.MSELoss()
    elif loss_func_aggr == 'l0.5':
        loss_func = lambda x, y: torch.mean(abs(x - y) ** 0.5)
    else:
        raise ValueError

    return loss_func


def sdf_loss(model, data, tmp, epoch, loss_func_aggr='l1', mask=None, coef=1.0):
    pred = model(data)
    loss_func_aggr = get_loss_func_aggr(loss_func_aggr)
    if mask is None:
        loss = loss_func_aggr(pred.x, pred.y)
    else:
        loss = loss_func_aggr(pred.x[mask], pred.y[mask])

    loss *= coef
    return loss


def get_numerical_diff(pred, model, data, dir, mask, data_parallel, eps):
    if data_parallel:
        data_copy = []
        for d, m in zip(data, mask):
            d_copy = d.clone()
            d_copy.x[m, dir] += eps
            d_copy.e = compute_edge_features(d_copy.x, d_copy.edge_index)
            data_copy.append(d_copy)
        mask = torch.cat(mask)
    else:
        data_copy = data.clone()
        data_copy.x[mask, dir] += eps
        data_copy.e = compute_edge_features(data_copy.x, data_copy.edge_index)

    dpred = model(data_copy)
    diff = (dpred.x[mask] - pred.x[mask]) / eps

    return diff


def normal_loss(model, data, data_normal, epoch, loss_func_aggr='l1', loss_func_aggr_norm='l1', coefs=None, eps=1e-4,
                every_epoch=None, min_epoch=None, coefs_factor=None):
    if every_epoch is not None and epoch % every_epoch != 0:
        return 0.

    if min_epoch is not None and epoch < min_epoch:
        return 0.

    if coefs is None:
        coefs = [1., 1.]

    if coefs_factor is not None:
        coefs = [min(coefs[0], coefs_factor[0] * epoch), min(coefs[1], coefs_factor[1] * epoch)]



    data_parallel = isinstance(data_normal, list)

    if data_parallel:# distributed data
        data_normal_copy = copy.deepcopy(data_normal)
        mask = []
        for d in data_normal_copy:
            mask.append(d.x[:, -1] == 0)
            d.x[:, -1] = 1
        pred = model(data_normal_copy)
        dpred_dx = get_numerical_diff(pred, model, data_normal_copy, 0, mask, data_parallel, eps)
        dpred_dy = get_numerical_diff(pred, model, data_normal_copy, 1, mask, data_parallel, eps)
        dpred_dz = get_numerical_diff(pred, model, data_normal_copy, 2, mask, data_parallel, eps)
    else:
        mask = data_normal.x[:, -1] == 0
        data_normal.x[:, -1] = 1
        pred = model(data_normal)
        dpred_dx = get_numerical_diff(pred, model, data_normal, 0, mask, data_parallel, eps)
        dpred_dy = get_numerical_diff(pred, model, data_normal, 1, mask, data_parallel, eps)
        dpred_dz = get_numerical_diff(pred, model, data_normal, 2, mask, data_parallel, eps)
    dpred = torch.cat((dpred_dx, dpred_dy, dpred_dz), dim=1)
    loss_func_aggr = get_loss_func_aggr(loss_func_aggr)
    loss_func_aggr_norm = get_loss_func_aggr(loss_func_aggr_norm)
    loss1 = loss_func_aggr(pred.x, torch.zeros_like(pred.x))
    if data_parallel:
        ys = torch.cat([d.y for d in data_normal]).to(device=dpred.device)
        loss2 = loss_func_aggr_norm(dpred, ys)
    else:
        loss2 = loss_func_aggr_norm(dpred, data_normal.y)
    loss = coefs[0] * loss1 + coefs[1] * loss2
    return loss


def get_loss_funcs(loss_funcs):
    LOSS_FUNC_NAME_DICT = {'sdf_loss': sdf_loss,
                           'normal_loss': normal_loss}

    if loss_funcs is None:
        loss_funcs = {'sdf_loss': {}}

    def compiled_loss_func(model, data, *args):
        losses = {}
        for loss_func, loss_funcs_params in loss_funcs.items():
            f = LOSS_FUNC_NAME_DICT[loss_func]
            losses[loss_func] = f(model, data, *args, **loss_funcs_params)

        return losses

    return compiled_loss_func


def get_optimizer(model, optimizer, lr_0):
    if optimizer.lower() == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr_0)
    else:
        raise ValueError
    return optim


def get_scheduler(optim, scheduler_type="StepLR", step_size=1000, gamma=0.2):
    if scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)
    else:
        raise ValueError
    return scheduler


def get_clip_gradients_func(clip_gradient_params):
    if clip_gradient_params is None:
        return

    if clip_gradient_params['type'] == 'clip_grad_value':
        clip_value = clip_gradient_params['clip_value']

        def clip_gradient_func(model):
            return torch.nn.utils.clip_grad_value_(model, clip_value)

    else:
        raise ValueError

    return clip_gradient_func


def get_summary_writer(save_folder_name):
    save_folder_name = os.path.join(os.getcwd(), save_folder_name)
    if not os.path.isdir(save_folder_name):
        os.makedirs(save_folder_name)
    tf_writer = SummaryWriter(save_folder_name)
    return tf_writer


def save_latest(model, epoch, optimizer, save_dir, data_parallel):
    # save lastet model
    model_latest_path = os.path.join(os.getcwd(), save_dir, "model_latest.pth")
    if data_parallel:
        torch.save({"epoch": epoch, "model_state_dict": model.module.state_dict()}, model_latest_path)
    else:
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, model_latest_path)

    # save latest optimizer
    optim_latest_path = os.path.join(os.getcwd(), save_dir, "optimizer_latest.pth")
    torch.save({"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()}, optim_latest_path)


def write_to_screen(epoch, optimizer, train_losses, test_losses=None):
    loss_names = train_losses[0].keys()
    tb_scalars_dict = {loss_name: torch.tensor([e[loss_name] for e in train_losses]).mean(dim=0)
                       for loss_name in loss_names}
    lr = optimizer.param_groups[0]['lr']
    print("epoch %4s: learning rate=%0.2e" % (str(epoch), lr), end="")
    for loss_name, loss_value in tb_scalars_dict.items():
        print(", train %s: %0.4e" % (loss_name, loss_value.item()), end="")

    if test_losses is not None and len(test_losses) > 0:
        tb_scalars_dict = {loss_name: torch.tensor([e[loss_name] for e in test_losses]).mean(dim=0)
                           for loss_name in loss_names}
        for loss_name, loss_value in tb_scalars_dict.items():
            print(", test %s: %0.4e" % (loss_name, loss_value.item()), end="")
    print(".")


def write_to_tensorboard(epoch, epoch_losses, tf_writer, tag):
    loss_names = epoch_losses[0].keys()
    tb_scalars_dict = {loss_name: torch.tensor([e[loss_name] for e in epoch_losses]).mean(dim=0)
                       for loss_name in loss_names}
    tf_writer.add_scalars(tag, tb_scalars_dict, epoch)


def write_gradients_to_file(named_parameters, epoch, save_folder_name):
    """
    https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    """
    ave_grads = []
    max_grads = []
    layers = []
    plt.figure(figsize=(6, 6))
    for n, p in named_parameters:
        if (p.requires_grad) and ('bias' not in n):
            layers.append(n.replace('weight', ''))
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.6, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.4, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.gca().tick_params(axis='x', direction="in")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=1.0)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()

    grads_dir = os.path.join(save_folder_name, "grads")
    if not os.path.isdir(grads_dir):
        os.makedirs(grads_dir)
    save_name = os.path.join(grads_dir, "grads_%d.jpg" % epoch)
    plt.savefig(save_name)
    plt.close()
