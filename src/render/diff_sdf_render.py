import torch
import renderer


def central_diff(x, dim):
    """
    this function compute the a central difference along dimension dim.
    dx_i = (x_(i+1) - x(i-1)) / 2
    for the edges, a forward method is used
    dx_0 = (-3x_0 + 4x_1 - x_2) / 2
    dx_N = (3x_N - 4x_(N-1) + x_(N-2)) / 2
    """
    n = x.size(dim)
    x_1_to_n = x.narrow(dim, 1, n - 1)
    x_0_to_n_minus_1 = x.narrow(dim, 0, n - 1)
    x_0 = x.narrow(dim, 0, 1)
    x_1 = x.narrow(dim, 1, 1)
    x_2 = x.narrow(dim, 2, 1)
    x_n = x.narrow(dim, n - 1, 1)
    x_n_minus_1 = x.narrow(dim, n - 2, 1)
    x_n_minus_2 = x.narrow(dim, n - 3, 1)

    right = torch.cat((x_1_to_n, 3 * x_n - 3 * x_n_minus_1 + x_n_minus_2), dim=dim)
    left = torch.cat((3 * x_0 - 3 * x_1 + x_2, x_0_to_n_minus_1), dim=dim)
    diff = (right - left) / 2
    return diff


def get_grid_normal(x, delta):
    dx = central_diff(x, 0) / delta
    dy = central_diff(x, 1) / delta
    dz = central_diff(x, 2) / delta
    return dx, dy, dz


def get_intersection_normal(intersection_grid_normal, intersection_pos, voxel_min_point, voxel_size):
    tx = (intersection_pos[:, :, 0] - voxel_min_point[:, :, 0]) / voxel_size
    ty = (intersection_pos[:, :, 1] - voxel_min_point[:, :, 1]) / voxel_size
    tz = (intersection_pos[:, :, 2] - voxel_min_point[:, :, 2]) / voxel_size

    intersection_normal = (1 - tz) * (1 - ty) * (1 - tx) * intersection_grid_normal[:, :, 0] + \
                          tz * (1 - ty) * (1 - tx) * intersection_grid_normal[:, :, 1]       + \
                          (1 - tz) * ty * (1 - tx) * intersection_grid_normal[:, :, 2]       + \
                          tz * ty * (1 - tx) * intersection_grid_normal[:, :, 3]             + \
                          (1 - tz) * (1 - ty) * tx * intersection_grid_normal[:, :, 4]       + \
                          tz * (1 - ty) * tx * intersection_grid_normal[:, :, 5]             + \
                          (1 - tz) * ty * tx * intersection_grid_normal[:, :, 6]             + \
                          tz * ty * tx * intersection_grid_normal[:, :, 7]

    return intersection_normal


def compute_intersection_pos(grid, intersection_pos_rough, voxel_min_point,
                             voxel_min_point_index, ray_direction, voxel_size, mask, width, height):

    # Linear interpolate along x axis the eight values
    tx = (intersection_pos_rough[:, :, 0] - voxel_min_point[:, :, 0]) / voxel_size

    ix = voxel_min_point_index.long()[:, :, 0]
    iy = voxel_min_point_index.long()[:, :, 1]
    iz = voxel_min_point_index.long()[:, :, 2]

    c01 = (1 - tx) * grid[ix, iy, iz] + tx * grid[ix+1, iy, iz]
    c23 = (1 - tx) * grid[ix, iy+1, iz] + tx * grid[ix+1, iy+1, iz]
    c45 = (1 - tx) * grid[ix, iy, iz+1] + tx * grid[ix+1, iy, iz+1]
    c67 = (1 - tx) * grid[ix, iy+1, iz+1] + tx * grid[ix+1, iy+1, iz+1]

    # Linear interpolate along the y axis
    ty = (intersection_pos_rough[:, :, 1] - voxel_min_point[:, :, 1]) / voxel_size
    c0 = (1 - ty) * c01 + ty * c23
    c1 = (1 - ty) * c45 + ty * c67

    # Return final value interpolated along z
    tz = (intersection_pos_rough[:, :, 2] - voxel_min_point[:, :, 2]) / voxel_size

    sdf_value = (1 - tz) * c0 + tz * c1

    intersection_pos = intersection_pos_rough + \
                       ray_direction * sdf_value.view(width,height, 1).repeat(1, 1, 3) + \
                       1 - mask.view(width, height, 1).repeat(1, 1, 3)
    return intersection_pos


def get_intersection_normal_helper(dsdf, ix, iy, iz, n, final_shape):
    n_times_iy = n * iy
    n2_times_ix = n ** 2 * ix
    x1 = torch.index_select(dsdf, 0, iz + n_times_iy + n2_times_ix).view(final_shape)
    x2 = torch.index_select(dsdf, 0, (iz + 1) + n_times_iy + n2_times_ix).view(final_shape)
    x3 = torch.index_select(dsdf, 0, iz + n_times_iy + n + n2_times_ix).view(final_shape)
    x4 = torch.index_select(dsdf, 0, (iz + 1) + n_times_iy + n + n2_times_ix).view(final_shape)
    x5 = torch.index_select(dsdf, 0, iz + n_times_iy + n2_times_ix + n ** 2).view(final_shape)
    x6 = torch.index_select(dsdf, 0, (iz + 1) + n_times_iy + n2_times_ix + n ** 2).view(final_shape)
    x7 = torch.index_select(dsdf, 0, iz + n_times_iy + n + n2_times_ix + n ** 2).view(final_shape)
    x8 = torch.index_select(dsdf, 0, (iz + 1) + n_times_iy + n + n2_times_ix + n ** 2).view(final_shape)
    out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), 2)
    return out


def check_inputs(sdf_grid, box):
    assert sdf_grid.get_device() >= 0, "the rendering only works with GPU."

    ndims = sdf_grid.dim()
    assert ndims <= 3, "sdf grid is a %d dimensional tensor, can't be converted to grid-like data" % (ndims)
    if ndims == 1:
        len_sdf = len(sdf_grid)
        grid_res = round(len_sdf ** (1 / 3))
        assert grid_res ** 3 == len_sdf, "sdf grid is of length %d. can't be converted to grid_like data" % (len_sdf)
    elif ndims == 2:
        len_sdf, s2 = sdf_grid.shape
        grid_res = round(len_sdf ** (1 / 3))
        assert s2 == 1, "sdf grid is of shape (%d, %d). can't be converted to grid_like data"% (len_sdf, s2)
        assert grid_res ** 3 == len_sdf, "sdf grid is of length %d. can't be converted to grid_like data" % (len_sdf)
    elif ndims == 3:
        grid_res_x, grid_res_y, grid_res_z = sdf_grid.shape
        assert grid_res_x == grid_res_y == grid_res_z, "renderer only supports cubic grids"

    if box is not None:
        bounding_box_min_x, bounding_box_min_y, bounding_box_min_z = box[0]
        bounding_box_max_x, bounding_box_max_y, bounding_box_max_z = box[1]
        assert bounding_box_max_x - bounding_box_min_x == bounding_box_max_z - bounding_box_min_z, "only supports cubic grids"
        assert bounding_box_max_x - bounding_box_min_x == bounding_box_max_y - bounding_box_min_y, "only supports cubic grids"


def render_surface_img(sdf_grid, camera_pos=None, box=None, img_size=None):
    check_inputs(sdf_grid, box)
    device = sdf_grid.device
    if img_size is None:
        width, height = 128, 128
    else:
        width, height = img_size

    if box is None:
        bounding_box_min_x, bounding_box_min_y, bounding_box_min_z = -1.0, -1.0, -1.0
        bounding_box_max_x, bounding_box_max_y, bounding_box_max_z = 1.0, 1.0, 1.0
    else:
        bounding_box_min_x, bounding_box_min_y, bounding_box_min_z = box[0]
        bounding_box_max_x, bounding_box_max_y, bounding_box_max_z = box[1]

    if camera_pos is None:
        camera_pos = torch.rand(3, device=device) - 0.5  # generate randomly between [-0.5, 0.5]
        camera_pos *= 2 / camera_pos.norm()  # making sure camera is outside object.
    else:
        camera_pos = torch.tensor(camera_pos, device=device)



    if sdf_grid.dim() <= 2:
        grid_res = round(sdf_grid.size(0) ** (1/3))
        sdf_grid = sdf_grid.view(grid_res, grid_res, grid_res)

    camera_x, camera_y, camera_z = camera_pos

    grid_res_x, grid_res_y, grid_res_z = sdf_grid.shape

    voxel_size = (bounding_box_max_x - bounding_box_min_x) / (grid_res_x - 1)

    # Get normal vectors for points on the grid
    dsdf_dx, dsdf_dy, dsdf_dz = get_grid_normal(sdf_grid, voxel_size)
    dsdf_dx, dsdf_dy, dsdf_dz = dsdf_dx.view(-1), dsdf_dy.view(-1), dsdf_dz.view(-1)

    # Do ray tracing in cpp
    w_h = torch.zeros(width, height, device=device)
    w_h_3 = torch.zeros(width, height, 3, device=device)
    intersection_pos_rough, voxel_min_point_index, ray_direction = \
        renderer.ray_matching(w_h_3, w_h, sdf_grid, width, height,
                              bounding_box_min_x, bounding_box_min_y, bounding_box_min_z,
                              bounding_box_max_x, bounding_box_max_y, bounding_box_max_z,
                              grid_res_x, grid_res_y, grid_res_z,
                              camera_x, camera_y, camera_z)

    # Make the pixels with no intersections with rays be 0
    mask = (voxel_min_point_index[:, :, 0] != -1).type(torch.float32)

    # Get the indices of the minimum point of the intersecting voxels
    ix = voxel_min_point_index[:, :, 0].type(torch.long).view(-1)
    iy = voxel_min_point_index[:, :, 1].type(torch.long).view(-1)
    iz = voxel_min_point_index[:, :, 2].type(torch.long).view(-1)
    ix[ix == -1] = 0
    iy[iy == -1] = 0
    iz[iz == -1] = 0

    # Get the x-axis of normal vectors for the 8 points of the intersecting voxel
    # This line is equivalent to grid_normal_x[x,y,z]
    final_shape = (width, height, 1)
    intersection_grid_normal_x = get_intersection_normal_helper(dsdf_dx, ix, iy, iz, grid_res_x, final_shape)
    intersection_grid_normal_x = intersection_grid_normal_x + 1 - mask.view(final_shape).repeat(1, 1, 8)

    # Get the y-axis of normal vectors for the 8 points of the intersecting voxel
    intersection_grid_normal_y = get_intersection_normal_helper(dsdf_dy, ix, iy, iz, grid_res_x, final_shape)
    intersection_grid_normal_y = intersection_grid_normal_y + 1 - mask.view(final_shape).repeat(1, 1, 8)

    # Get the z-axis of normal vectors for the 8 points of the intersecting voxel
    intersection_grid_normal_z = get_intersection_normal_helper(dsdf_dz, ix, iy, iz, grid_res_x, final_shape)
    intersection_grid_normal_z = intersection_grid_normal_z + 1 - mask.view(final_shape).repeat(1, 1, 8)

    # Change from grid coordinates to world coordinates
    voxel_min_point = torch.tensor([bounding_box_min_x, bounding_box_min_y, bounding_box_min_z], device=device)
    voxel_min_point = voxel_min_point + voxel_min_point_index * voxel_size

    intersection_pos = compute_intersection_pos(sdf_grid, intersection_pos_rough,
                                                voxel_min_point, voxel_min_point_index,
                                                ray_direction, voxel_size, mask, width, height)

    intersection_pos = intersection_pos * mask.repeat(3, 1, 1).permute(1, 2, 0)

    # Compute the normal vectors for the intersecting points
    intersection_normal_x = get_intersection_normal(intersection_grid_normal_x, intersection_pos, voxel_min_point,
                                                    voxel_size)
    intersection_normal_y = get_intersection_normal(intersection_grid_normal_y, intersection_pos, voxel_min_point,
                                                    voxel_size)
    intersection_normal_z = get_intersection_normal(intersection_grid_normal_z, intersection_pos, voxel_min_point,
                                                    voxel_size)

    # Put all the xyz-axis of the normal vectors into a single matrix
    intersection_normal = torch.stack((intersection_normal_x, intersection_normal_y, intersection_normal_z), 2)
    intersection_normal_norm = torch.norm(intersection_normal, keepdim=True, dim=2).repeat(1, 1, 3)
    intersection_normal = intersection_normal / intersection_normal_norm

    # Create the point light
    light_position = camera_pos.repeat(width, height, 1)
    light_norm = torch.norm(light_position - intersection_pos, keepdim=True, dim=2).repeat(1, 1, 3)
    light_direction_point = (light_position - intersection_pos) / light_norm

    # Create the directional light
    light_direction = (camera_pos / torch.norm(camera_pos)).repeat(width, height, 1)
    l_dot_n = torch.sum(light_direction * intersection_normal, dim=2, keepdim=True)

    numer = torch.max(l_dot_n, torch.zeros(width, height, 1, device=device))[:, :, 0]
    denom = torch.sum((light_position - intersection_pos) * light_direction_point, dim=2) ** 2
    shading = 10 * numer / denom

    image = shading * mask
    image[mask == 0] = 0

    return image
