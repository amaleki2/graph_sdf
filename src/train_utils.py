import os
import sys
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


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


def graph_loss(data, loss_func=torch.nn.L1Loss()):
    loss = loss_func(data.x, data.y)
    return loss


def graph_loss_data_parallel(data, loss_func=torch.nn.L1Loss()):
    device = data.device
    data_y = torch.cat([d.y for d in data]).to(device)
    loss = loss_func(data.x, data_y)
    return loss


def get_loss_func(loss_func, data_parallel):
    if loss_func == 'l1':
        func = torch.nn.L1Loss()
    elif loss_func == 'l2':
        func = torch.nn.L1Loss()
    else:
        raise ValueError

    if data_parallel:
        return lambda x: graph_loss_data_parallel(x, loss_func=func)
    else:
        return lambda x: graph_loss(x, loss_func=func)


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
    print(", train loss: ", end="")
    print('%.4f' % train_loss, end="")
    if test_loss:
        print(", test loss: ", end="")
        print('%.4f' % test_loss)
    else:
        print("")