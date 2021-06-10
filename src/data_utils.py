import trimesh
import numpy as np
from sklearn.neighbors import KDTree
from trimesh.proximity import ProximityQuery


def transform_mesh(mesh, trans_name, trans_params):
    if trans_name == 'preprocess':
        mesh = preprocess_mesh(mesh, **trans_params)
    elif trans_name == 'refine':
        mesh = refine_mesh(mesh, **trans_params)
    elif trans_name == 'rotate':
        mesh = rotate_mesh(mesh, **trans_params)
    else:
        raise ValueError("transformation type %d is not defined"%trans_name)

    return mesh


def preprocess_mesh(mesh,
                    mesh_out=None,
                    merge_vertex=True,
                    with_scaling_to_unit_box=True,
                    scaler=2):
    """
    process the mesh to ensure it is watertight and fits a unit cube [-1,1]^3
    """

    if merge_vertex:
        mesh.merge_vertices(merge_tex=True, merge_norm=True)

    if not mesh.is_watertight:
        raise ValueError('mesh is not watertight')

    if with_scaling_to_unit_box:
        s1 = mesh.bounding_box.centroid
        s2 = scaler / np.max(mesh.bounding_box.extents)
        new_vertices = mesh.vertices - s1
        mesh.vertices = new_vertices * s2

    if mesh_out is not None:
        with open(mesh_out, 'w') as fid:
            mesh.export(fid, file_type='obj')

    return mesh


def refine_mesh(mesh,
                mesh_out=None,
                mesh_refine_size=0.1,
                show=False):
    """
    generate refined surface mesh
    """
    refined_mesh = refine_surface_mesh(mesh, mesh_size=mesh_refine_size, show=show)

    if mesh_out is not None:
        with open(mesh_out, 'w') as fid:
            refined_mesh.export(fid, file_type='obj')

    return refined_mesh


def rotate_mesh(mesh,
                matrix=None):
    if matrix is None:
        matrix = trimesh.transformations.random_rotation_matrix()
    mesh.apply_transform(matrix)
    return mesh


def get_volume_points_randomly(n_points, scaler=2):
    points = np.random.random((n_points, 3)) - 0.5
    points *= scaler
    return points


def get_sdf(mesh, points):
    return - ProximityQuery(mesh).signed_distance(points)

# def remove_volume_edges(node_attr, edge_idx):
#     on_surface_idx = np.where(node_attr[:, -1] == 1)[0]
#     mask = np.isin(edge_idx, on_surface_idx).any(axis=0)
#     new_edge_idx = edge_idx[:, mask]
#     return new_edge_idx


def ball_query(x1, x2, radius=0.1, min_n_edges=3, max_n_edges=50):
    tree = KDTree(x2)
    dist, idx = tree.query(x1, k=max_n_edges)
    s1, s2 = idx.shape
    idx = np.stack((np.tile(np.arange(s1), (s2, 1)).T, idx), axis=2).reshape(-1, 2)  # get list of pairs
    indicator = dist < radius
    indicator[:, :min_n_edges] = 1  # set the minimum number of edges
    indicator = indicator.reshape(-1)
    idx = idx[indicator]  # set the radius of proximity
    edges = idx.T
    return edges


def get_edges_with_ball_query(x, radius=0.1, min_n_edges=3, max_n_edges=50, n_features_to_consider=3,
                              with_volume_edges=True):
    points = x[:, :n_features_to_consider]
    if with_volume_edges:
        edges = ball_query(points, points, radius=radius, min_n_edges=min_n_edges, max_n_edges=max_n_edges)
    else:
        sdf_indicator = x[:, -1]
        surface_points = points[sdf_indicator == 1]
        volume_points = points[sdf_indicator != 1]
        edges1 = ball_query(surface_points, points, radius=radius, min_n_edges=min_n_edges, max_n_edges=max_n_edges)
        edges2 = ball_query(volume_points, surface_points, radius=radius, min_n_edges=min_n_edges, max_n_edges=max_n_edges)
        edges2[0] = edges2[0] + len(surface_points)
        edges = np.concatenate((edges1, edges2), axis=1)
    return edges


def add_reversed_edges(edges):
    edges_reversed = np.flipud(edges)
    edges = np.concatenate([edges, edges_reversed], axis=1)
    return edges


def add_self_edges(edges):
    n_nodes = edges.max() + 1
    self_edges = [list(range(n_nodes))] * 2
    self_edges = np.array(self_edges)
    edges = np.concatenate([edges, self_edges], axis=1)
    return edges


def compute_edge_features(x, edge_index):
    e1, e2 = edge_index
    edge_attrs = x[e1, :] - x[e2, :]
    return edge_attrs


def get_mesh_edges(mesh):
    return mesh.edges.T


