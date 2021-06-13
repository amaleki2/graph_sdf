import os
import tqdm
import torch
from joblib import Parallel, delayed
from torch_geometric.data import Data, DataLoader, DataListLoader

from .data_utils import *


class SDF3dData:
    def __init__(self,
                 node_params=None,
                 edge_params=None,
                 global_params=None,
                 transform_params=None,
                 dataloader_params=None,
                 validate_args=True,
                 ):

        if validate_args:
            assert node_params is not None, "node parameters should be specified"
            assert edge_params is not None, "edge parameters should be specified"

        self.node_params       = node_params
        self.edge_params       = edge_params
        self.global_params     = global_params
        self.transform_params  = transform_params
        self.dataloader_params = dataloader_params

    @staticmethod
    def _get_node_attr(mesh, n_volume_points=None, on_surface_only=True, scaler=2):
        if n_volume_points is None:
            n_volume_points = np.random.randint(1000, 10000)

        surface_points = mesh.vertices
        volume_points = get_volume_points_randomly(n_volume_points, scaler=scaler)
        volume_sdfs = get_sdf(mesh, volume_points)
        all_points = np.concatenate((surface_points, volume_points))
        all_sdfs = np.concatenate((np.zeros(len(surface_points)), volume_sdfs))

        if on_surface_only:
            additional_feature = (all_sdfs == 0).reshape(-1, 1)
        else:
            additional_feature = (all_sdfs < 0).reshape(-1, 1)

        x = np.concatenate((all_points, additional_feature), axis=1)
        x = x.astype(float)
        y = all_sdfs.reshape(-1, 1).astype(float)
        return x, y

    @staticmethod
    def _get_edge_attr(mesh,
                       node_attr,
                       edge_method='mesh_edge',
                       radius=0.25,
                       min_n_edges=3,
                       max_n_edges=25,
                       n_features_to_consider=3,
                       include_reverse_edges=False,
                       include_self_edges=False,
                       with_volume_edges=True):

        if edge_method == 'mesh_edge':
            edge_idx = get_mesh_edges(mesh)
        elif edge_method == 'ball_query':
            edge_idx = get_edges_with_ball_query(node_attr,
                                                 radius=radius,
                                                 min_n_edges=min_n_edges,
                                                 max_n_edges=max_n_edges,
                                                 n_features_to_consider=n_features_to_consider,
                                                 with_volume_edges=with_volume_edges)
        elif edge_method == 'both':
            edges1 = get_mesh_edges(mesh)
            edges2 = get_edges_with_ball_query(node_attr,
                                               radius=radius,
                                               min_n_edges=min_n_edges,
                                               max_n_edges=max_n_edges,
                                               n_features_to_consider=n_features_to_consider,
                                               with_volume_edges=with_volume_edges)
            edge_idx = np.concatenate((edges1, edges2), axis=1)
        else:
            raise (NotImplementedError("method %s is not recognized" % edge_method))
        # edges = edges.T

        if include_reverse_edges:
            edge_idx = add_reversed_edges(edge_idx)

        if include_self_edges:
            edge_idx = add_self_edges(edge_idx)

        edge_idx = np.unique(edge_idx, axis=1)  # remove repeated edges

        edge_attr = compute_edge_features(node_attr, edge_idx)

        return edge_attr, edge_idx

    @staticmethod
    def _get_global_attr(mesh, centroid=True, volume=True, area=True):
        global_attr = []
        if centroid:
            global_attr += mesh.bounding_box.centroid.tolist()
        if area:
            global_attr.append(mesh.area)
        if volume:
            global_attr.append(mesh.volume)
        global_attr = np.array(global_attr).reshape(1, -1)
        return global_attr

    def _mesh_to_dataloader(self,
                            n_objects=None,
                            randomize_objects=False,
                            data_folder=None,
                            n_workers=1,
                            data_parallel=False,
                            batch_size=2,
                            shuffle_dataloader=False,
                            eval_frac=0.1,
                            data_filter=None):

        if data_filter is None:
            data_filter = lambda x: x.endswith('.obj')  # return true always

        files = os.listdir(data_folder)
        files = [os.path.join(data_folder, f) for f in files if data_filter(f)]
        files = np.array(files)  # fir the ease of slicing

        if n_objects is None:
            n_objects = len(files)
        else:
            if randomize_objects:
                idxs = np.random.randint(0, len(files), n_objects)
            else:
                idxs = np.arange(n_objects)
            files = files[idxs]

        n_train = round((1 - eval_frac) * n_objects)
        train_idx = np.arange(n_train)
        test_idx = np.arange(n_train, n_objects)
        train_files = files[train_idx]
        test_files = files[test_idx]

        if n_workers == 1:
            train_graph_data_list = [self.mesh_to_graph(f) for f in tqdm.tqdm(train_files)]
            test_graph_data_list = [self.mesh_to_graph(f) for f in tqdm.tqdm(test_files)]
        else:
            train_graph_data_list = Parallel(n_jobs=n_workers)(delayed(self.mesh_to_graph)(f)
                                                               for f in tqdm.tqdm(train_files))
            test_graph_data_list = Parallel(n_jobs=n_workers)(delayed(self.mesh_to_graph)(f)
                                                              for f in tqdm.tqdm(test_files))

        if data_parallel:
            train_data = DataListLoader(train_graph_data_list, batch_size=batch_size, shuffle=shuffle_dataloader)
            test_data = DataListLoader(test_graph_data_list, batch_size=batch_size, shuffle=shuffle_dataloader)
        else:
            train_data = DataLoader(train_graph_data_list, batch_size=batch_size, shuffle=shuffle_dataloader)
            test_data = DataLoader(test_graph_data_list, batch_size=batch_size, shuffle=shuffle_dataloader)
        return train_data, test_data

    def mesh_to_graph(self, mesh_file):
        """
        read mesh and generate pyg Data
        """
        mesh = trimesh.load(mesh_file, force='mesh', skip_materials=True)

        if self.transform_params is not None:
            for transform_name, transform_params in self.transform_params.items():
                mesh = transform_mesh(mesh, transform_name, transform_params)

        node_attr, node_sdf = self._get_node_attr(mesh, **self.node_params)
        edge_attr, edge_idx = self._get_edge_attr(mesh, node_attr, **self.edge_params)

        if self.global_params is None:
            pyg_data = Data(x=torch.from_numpy(node_attr).type(torch.float32),
                            y=torch.from_numpy(node_sdf).type(torch.float32),
                            e=torch.from_numpy(edge_attr).type(torch.float32),
                            edge_index=torch.from_numpy(edge_idx).type(torch.long))
        else:
            global_attr = self._get_global_attr(mesh, **self.global_params)
            pyg_data = Data(x=torch.from_numpy(node_attr).type(torch.float32),
                            y=torch.from_numpy(node_sdf).type(torch.float32),
                            e=torch.from_numpy(edge_attr).type(torch.float32),
                            u=torch.from_numpy(global_attr).type(torch.float32),
                            edge_index=torch.from_numpy(edge_idx).type(torch.long))
        return pyg_data

    def mesh_to_dataloader(self):
        return self._mesh_to_dataloader(**self.dataloader_params)