import os
import sys
import json
import argparse

from src import SDF3dData, train_sdf

parent_folder = os.path.split(os.path.dirname(__file__))[0]
sys.path.append(parent_folder)
from graph_networks.src import EncodeProcessDecode


class ExampleSDFTraining:
    def __init__(self, example_folder):

        os.chdir(example_folder)

        with open("network_configs.json", "rb") as fid:
            self.network_configs = json.load(fid)

        with open("data_configs.json", "rb") as fid:
            self.data_configs = json.load(fid)

        with open("training_configs.json", "rb") as fid:
            self.training_configs = json.load(fid)

    def get_network(self):
        model_params = self.network_configs['encode_process_decode']
        model = EncodeProcessDecode(**model_params)
        return model

    def get_dataloader(self):
        data_params = self.data_configs['SDF3dData']
        data_handler = SDF3dData(**data_params)
        train_dl, test_dl = data_handler.mesh_to_dataloader()
        return train_dl, test_dl

    def train(self):
        train_params = self.training_configs['train']
        model = self.get_network()
        train_dataloader, test_dataloader = self.get_dataloader()
        train_sdf(model, train_dataloader, test_dataloader, **train_params)

    def test(self):
        pass


parser = argparse.ArgumentParser(description='SDF Graph')
parser.add_argument('-e', dest='example_folder', type=str, required=True)
parser.add_argument('-test', dest='test', type=bool)
parser = parser.parse_args()

example = ExampleSDFTraining(parser.example_folder)
if parser.test:
    example.test()
else:
    example.train()