import json
import shutil
import unittest
from src import SDF3dData, train_sdf
from src.train_utils import *


class SDFTrainTest(unittest.TestCase):
    @staticmethod
    def get_abs_path():
        path = os.path.abspath(__file__)
        parent_dir = os.path.split(path)[0]
        return parent_dir

    def get_dataloaders(self):
        parent_dir = self.get_abs_path()
        data_configs = os.path.join(parent_dir, "configs", "test_data_configs.json")
        with open(data_configs, "rb") as fid:
            configs = json.load(fid)
        configs = configs['train_data']
        data_handler = SDF3dData(**configs)
        train_dl, test_dl = data_handler.mesh_to_dataloader()
        return train_dl, test_dl

    def get_network(self):
        abs_path = os.path.abspath(__file__)
        parent_folder = os.path.split(os.path.split(os.path.split(abs_path)[0])[0])[0]
        sys.path.append(parent_folder)
        from graph_networks.src import EncodeProcessDecode

        parent_dir = self.get_abs_path()
        network_configs = os.path.join(parent_dir, "configs", "test_network_configs.json")
        with open(network_configs, "rb") as fid:
            config = json.load(fid)
        model_params = config['encode_process_decode']
        model = EncodeProcessDecode(**model_params)
        return model

    def test_graph_loss(self):
        loss_func = 'l1'
        data_parallel = False
        func = get_loss_funcs(loss_func, data_parallel)[0]
        train_dl, _ = self.get_dataloaders()
        data = next(iter(train_dl))
        data.x = torch.norm(data.x, dim=1) - 0.5
        func(data)

    def test_render_loss(self):
        loss_func = 'render_l1'
        data_parallel = False
        func = get_loss_funcs(loss_func, data_parallel)[0]
        train_dl, _ = self.get_dataloaders()
        data = next(iter(train_dl)).to(device='cuda')
        data.x = torch.norm(data.x, dim=1) - 0.5
        func(data)

    def test_train(self):
        parent_dir = self.get_abs_path()
        training_configs = os.path.join(parent_dir, "configs", "test_training_configs.json")
        with open(training_configs, "rb") as fid:
            configs = json.load(fid)
        training_params = configs['train']
        model = self.get_network()
        train_dataloader, test_dataloader = self.get_dataloaders()
        train_sdf(model, train_dataloader, test_dataloader, **training_params)
        save_dir = os.path.join(os.getcwd(), training_params['save_folder_name'])
        self.assertTrue(os.path.isdir(save_dir))
        shutil.rmtree(save_dir)


if __name__ == '__main__':
    unittest.main()