import os
import json
import trimesh
import unittest
import numpy as np
from src.data_utils import transform_mesh
from src import SDF3dData


class SDF3dDataTest(unittest.TestCase):
    @staticmethod
    def get_abs_path():
        path = os.path.abspath(__file__)
        parent_dir = os.path.split(path)[0]
        return parent_dir

    def read_cube_mesh(self):
        parent_dir = self.get_abs_path()
        mesh_file = os.path.join(parent_dir, "objects", "test_cube.obj")
        mesh = trimesh.load(mesh_file)
        mesh.vertices *= 0.5
        return mesh

    def test_transform(self):
        mesh = self.read_cube_mesh()
        mesh = transform_mesh(mesh, 'preprocess', {})
        mesh = transform_mesh(mesh, 'rotate', {})
        self.assertTrue(mesh.is_watertight)
        self.assertAlmostEqual(mesh.area, 24.0)
        self.assertAlmostEqual(mesh.volume, 8.0)

    def test_node_attr(self):
        data_handler = SDF3dData(validate_args=False)
        mesh = self.read_cube_mesh()
        x, y = data_handler._get_node_attr(mesh, n_volume_points=10, on_surface_only=True)
        self.assertEqual(len(x), 18)
        self.assertEqual((y == 0).sum(), 8)

        x, y = data_handler._get_node_attr(mesh, n_volume_points=10, on_surface_only=False)
        self.assertEqual(len(x), 18)

    def test_edge_attr(self):
        data_handler = SDF3dData(validate_args=False)
        mesh = self.read_cube_mesh()
        n_volume_points = 3
        x, y = data_handler._get_node_attr(mesh, n_volume_points=n_volume_points, on_surface_only=True, scaler=1.5)
        self.assertEqual(len(x), 8 + n_volume_points)

        edge_attr, edge_idx = data_handler._get_edge_attr(mesh, x,
                                                          edge_method='mesh_edge',
                                                          include_reverse_edges=False,
                                                          with_volume_edges=True)
        self.assertEqual(len(edge_attr), 36)

        edge_attr, edge_idx = data_handler._get_edge_attr(mesh, x[:8],
                                                          edge_method='ball_query',
                                                          radius=0.25,
                                                          min_n_edges=3,
                                                          max_n_edges=5,
                                                          n_features_to_consider=3,
                                                          include_reverse_edges=False,
                                                          with_volume_edges=True)
        self.assertEqual(len(edge_attr), 24)

        edge_attr, edge_idx = data_handler._get_edge_attr(mesh, x,
                                                          edge_method='ball_query',
                                                          radius=0.25,
                                                          min_n_edges=3,
                                                          max_n_edges=5,
                                                          n_features_to_consider=3,
                                                          include_reverse_edges=False,
                                                          include_self_edges=False,
                                                          with_volume_edges=False)

        # check no volume edge exists
        for i in range(8, 8 + n_volume_points):
            e = [x[0] for x in edge_idx.T if x[1] == i] + [x[1] for x in edge_idx.T if x[0] == i]
            self.assertTrue((np.array(e) < 8).all())

    def test_global_attr(self):
        data_handler = SDF3dData(validate_args=False)
        mesh = self.read_cube_mesh()
        u = data_handler._get_global_attr(mesh, centroid=True, volume=True, area=True)
        self.assertTrue(np.isclose(u, [0, 0, 0, 6, 1]).all())

    def test_SDF3dData(self):
        parent_dir = self.get_abs_path()
        data_configs = os.path.join(parent_dir, "configs", "test_data_configs.json")
        with open(data_configs, "rb") as fid:
            configs = json.load(fid)
        configs = configs['SDF3dData']
        data_handler = SDF3dData(**configs)
        data_handler.mesh_to_dataloader()


if __name__ == '__main__':
    unittest.main()