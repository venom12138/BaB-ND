import numpy as np
import open3d as o3d


def _init_low_level_memory(lower_bound, higher_bound, voxel_size, voxel_num):
    def pcd_to_voxel(pcds):
        if type(pcds) == list:
            pcds = np.array(pcds)
        # The pc is in numpy array with shape (..., 3)
        # The voxel is in numpy array with shape (..., 3)
        voxels = np.floor((pcds - lower_bound) / voxel_size).astype(np.int32)
        return voxels

    def voxel_to_pcd(voxels):
        if type(voxels) == list:
            voxels = np.array(voxels)
        # The voxel is in numpy array with shape (..., 3)
        # The pc is in numpy array with shape (..., 3)
        pcds = voxels * voxel_size + lower_bound
        return pcds

    def voxel_to_index(voxels):
        if type(voxels) == list:
            voxels = np.array(voxels)
        # The voxel is in numpy array with shape (..., 3)
        # The index is in numpy array with shape (...,)
        indexes = (
            voxels[..., 0] * voxel_num[1] * voxel_num[2]
            + voxels[..., 1] * voxel_num[2]
            + voxels[..., 2]
        )
        return indexes

    def index_to_voxel(indexes):
        if type(indexes) == list:
            indexes = np.array(indexes)
        # The index is in numpy array with shape (...,)
        # The voxel is in numpy array with shape (..., 3)
        voxels = np.zeros((indexes.shape + (3,)), dtype=np.int32)
        voxels[..., 2] = indexes % voxel_num[2]
        indexes = indexes // voxel_num[2]
        voxels[..., 1] = indexes % voxel_num[1]
        voxels[..., 0] = indexes // voxel_num[1]
        return voxels

    def pcd_to_index(pcds):
        # The pc is in numpy array with shape (..., 3)
        # The index is in numpy array with shape (...,)
        voxels = pcd_to_voxel(pcds)
        indexes = voxel_to_index(voxels)
        return indexes

    def index_to_pcd(indexes):
        # The index is in numpy array with shape (...,)
        # The pc is in numpy array with shape (..., 3)
        voxels = index_to_voxel(indexes)
        pcds = voxel_to_pcd(voxels)
        return pcds

    return (
        pcd_to_voxel,
        voxel_to_pcd,
        voxel_to_index,
        index_to_voxel,
        pcd_to_index,
        index_to_pcd,
    )


lower_bound = np.array([-1, -1,  -0.2, ])
higher_bound = np.array([1, 1, 1])

voxel_size = 0.01
voxel_num = ((higher_bound - lower_bound) / voxel_size).astype(np.int32)
(
    pcd_to_voxel,
    voxel_to_pcd,
    voxel_to_index,
    index_to_voxel,
    pcd_to_index,
    index_to_pcd,
) = _init_low_level_memory(
    lower_bound, higher_bound, voxel_size, voxel_num=voxel_num
)


def depth2fgpcd(depth, mask, cam_params):
    # depth: (h, w)
    # fgpcd: (n, 3)
    # mask: (h, w)
    h, w = depth.shape
    mask = np.logical_and(mask, depth > 0)
    # mask = (depth <= 0.599/0.8)
    fgpcd = np.zeros((mask.sum(), 3))
    fx, fy, cx, cy = cam_params
    pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
    fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
    fgpcd[:, 2] = depth[mask]
    return fgpcd

def np2o3d(pcd, color=None):
    # pcd: (n, 3)
    # color: (n, 3)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None:
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1
        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d

def aggr_point_cloud_from_data(colors, depths, Ks, poses, downsample=True, masks=None):
    # colors: [N, H, W, 3] numpy array in uint8
    # depths: [N, H, W] numpy array in meters
    # Ks: [N, 3, 3] numpy array
    # poses: [N, 4, 4] numpy array
    # masks: [N, H, W] numpy array in bool
    N, H, W, _ = colors.shape
    colors = colors / 255.
    start = 0
    end = N
    # end = 2
    step = 1
    # visualize scaled COLMAP poses
    pcds = []
    for i in range(start, end, step):
        depth = depths[i]
        color = colors[i]
        K = Ks[i]
        cam_param = [K[0,0], K[1,1], K[0,2], K[1,2]] # fx, fy, cx, cy
        if masks is None:
            mask = (depth > 0) & (depth < 1.5)
        else:
            mask = masks[i]
        # mask = np.ones_like(depth, dtype=bool)
        pcd = depth2fgpcd(depth, mask, cam_param)
        
        pose = poses[i]
        pose = np.linalg.inv(pose)
        
        trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
        trans_pcd = trans_pcd[:3, :].T
        
        # plt.subplot(1, 4, 1)
        # plt.imshow(trans_pcd[:, 0].reshape(H, W))
        # plt.subplot(1, 4, 2)
        # plt.imshow(trans_pcd[:, 1].reshape(H, W))
        # plt.subplot(1, 4, 3)
        # plt.imshow(trans_pcd[:, 2].reshape(H, W))
        # plt.subplot(1, 4, 4)
        # plt.imshow(color)
        # plt.show()
        
        pcd_o3d = np2o3d(trans_pcd, color[mask])
        # downsample
        if downsample:
            radius = 0.01
            pcd_o3d = pcd_o3d.voxel_down_sample(radius)
        pcds.append(pcd_o3d)
    aggr_pcd = o3d.geometry.PointCloud()
    for pcd in pcds:
        aggr_pcd += pcd
    return aggr_pcd




def aggr_raw_pc_from_data(depths, Ks, poses, masks=None):
    N, H, W = depths.shape
    start = 0
    end = N
    step = 1
    pcds = []
    for i in range(start, end, step):
        depth = depths[i]
        K = Ks[i]
        cam_param = [K[0,0], K[1,1], K[0,2], K[1,2]] # fx, fy, cx, cy
        if masks is None:
            mask = (depth > 0) & (depth < 1.5)
        else:
            mask = masks[i] & (depth > 0)
        # mask = np.ones_like(depth, dtype=bool)
        pcd = depth2fgpcd(depth, mask, cam_param)
        
        pose = poses[i]
        pose = np.linalg.inv(pose)
        
        trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
        trans_pcd = trans_pcd[:3, :].T
        
        pcd_np =  trans_pcd

        pcds.append(pcd_np)

    pcds = np.concatenate(pcds, axis=0)
        # pcd_colors = np.concatenate(pcd_colors, axis=0)
    return pcds
    


