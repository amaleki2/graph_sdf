import os
import sys
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .render.diff_sdf_render import render_surface_img


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


def graph_loss(data, loss_func=torch.nn.L1Loss(), data_parallel=False, mask=None):
    if not data_parallel:
        if mask is None:
            loss = loss_func(data.x, data.y)
        else:
            loss = loss_func(data.x[mask], data.y[mask])
    else:
        raise NotImplemented
    return loss


def graph_loss_banded(data, loss_func=torch.nn.L1Loss(), data_parallel=False, intervals=(0.01, 0.1), weights=(100, 10)):
    loss = graph_loss(data, loss_func=loss_func, data_parallel=data_parallel)
    for interval, weight in zip(intervals, weights):
        mask = abs(data.y) < interval
        added_loss = graph_loss(data, loss_func=loss_func, data_parallel=data_parallel, mask=mask)
        loss = weight * added_loss
    return loss


def graph_loss_render(data, loss_func=torch.nn.L1Loss(), camera_pos=None, data_parallel=False, box=None, img_size=None):
    # split batches
    if data_parallel:
        raise NotImplemented

    if camera_pos is None:
        camera_pos = torch.rand(3, device=data.x.device)

    loss_value = 0
    left_idx = 0
    batches = torch.cumsum(torch.bincount(data.batch), 0)
    for right_idx in batches:
        sdf_pred = data.x[left_idx:right_idx]
        sdf_truth = data.y[left_idx:right_idx]
        n_surface_nodes = (sdf_truth == 0).sum()  # we should skip the surface nodes. they have sdf=0.
        img_pred = render_surface_img(sdf_pred[n_surface_nodes:], camera_pos=camera_pos, box=box, img_size=img_size)
        img_truth = render_surface_img(sdf_truth[n_surface_nodes:], camera_pos=camera_pos, box=box, img_size=img_size)
        loss_value += loss_func(img_pred, img_truth)
        left_idx = right_idx

    return loss_value / len(batches)


def get_loss_funcs(loss_funcs, data_parallel):
    funcs = []
    if not isinstance(loss_funcs, list):
        loss_funcs = [loss_funcs]

    for loss_func in loss_funcs:
        if loss_func == 'l1':
            func = lambda x: graph_loss(x, loss_func=torch.nn.L1Loss(), data_parallel=data_parallel)
        elif loss_func == 'l2':
            func = lambda x: graph_loss(x, loss_func=torch.nn.MSELoss(), data_parallel=data_parallel)
        elif loss_func == 'banded_l1':
            func = lambda x: graph_loss_banded(x, loss_func=torch.nn.L1Loss(), data_parallel=data_parallel)
        elif loss_func == 'render_l1':
            func = lambda x: graph_loss_render(x, loss_func=torch.nn.L1Loss(), data_parallel=data_parallel)
        else:
            raise ValueError
        funcs.append(func)
    return funcs


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


def print_to_screen(epoch, optimizer, train_loss, test_loss=None):
    lr = optimizer.param_groups[0]['lr']
    print("epoch %4s: learning rate=%0.2e" % (str(epoch), lr), end="")
    for i, l in enumerate(train_loss):
        print(", train loss %d: %0.4f" % (i, l.item()), end="")
    if test_loss:
        for i, l in enumerate(test_loss):
            print(", test loss %d: %0.4f" % (i, l.item()), end="")
    print("")