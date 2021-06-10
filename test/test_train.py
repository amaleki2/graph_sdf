import json
import shutil
import unittest
from src import SDF3dData, train_sdf
from src.train_utils import *

parent_folder = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]
sys.path.append(parent_folder)
from graph_networks.src import EncodeProcessDecode


class SDFTrainTest(unittest.TestCase):
    @staticmethod
    def get_dataloaders():
        with open("configs/test_data_configs.json", "rb") as fid:
            configs = json.load(fid)
        configs = configs['SDF3dData']
        data_handler = SDF3dData(**configs)
        train_dl, test_dl = data_handler.mesh_to_dataloader()
        return train_dl, test_dl

    @staticmethod
    def get_network():
        with open("configs/test_network_configs.json", "rb") as fid:
            config = json.load(fid)
        model_params = config['encode_process_decode']
        model = EncodeProcessDecode(**model_params)
        return model

    def test_graph_loss(self):
        loss_func = 'l1'
        data_parallel = False
        func = get_loss_func(loss_func, data_parallel)
        train_dl, _ = self.get_dataloaders()
        data = next(iter(train_dl))
        data.x = data.x[:, :1]
        func(data)

    def test_train(self):
        with open("configs/test_training_configs.json", "rb") as fid:
            configs = json.load(fid)
        training_params = configs['train']
        model = self.get_network()
        train_dataloader, test_dataloader = self.get_dataloaders()
        train_sdf(model, train_dataloader, test_dataloader, **training_params)
        save_dir = os.path.join(os.getcwd(), training_params['save_folder_name'])
        self.assertTrue(os.path.isdir(save_dir))
        shutil.rmtree(save_dir)