import numpy as np
import open3d as o3d

def show_depth_image_matplotlib(pointcloud_path):
    points =np.load(pointcloud_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries([pcd],window_name="ShowCloudPoint")

if __name__ == '__main__':
    pointcloud_path = ''
    show_depth_image_matplotlib(pointcloud_path)
