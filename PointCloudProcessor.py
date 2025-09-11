import open3d as o3d
import numpy as np
import cv2

# class CloudPointProcessor:
#     # def __init__(self,intrinsics):
#     #     """初始化点云处理器"""
#     #     self.fx = intrinsics["fx"]
#     #     self.fy = intrinsics["fy"]
#     #     self.cx = intrinsics["cx"]
#     #     self.cy = intrinsics["cy"]
#     #     self.depth_scale = intrinsics["depth_scale",0.001]  # 深度单位转换（mm→m）

# 体素网格降采样
def reduce_pointcloud(points):
    if len(points) >0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        voxel_size = 0.05   # 体素大小（单位：米，值越大点数越少）
        downsample_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        downsample_points = np.asarray(downsample_pcd.points)
        return downsample_points
    else:
        return points


