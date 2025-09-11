import os

import cv2
import open3d as o3d

import PointCloudProcessor
import pykinect_azure as pykinect
from TwoStageDeployer import TwoStageDeployer
from utils import Open3dVisualizer
from PointCloudProcessor import *


class Run:
    def __init__(self,
                 _device_config,
                 color_format,
                 color_resolution,
                 depth_mode,
                 primary_model_path=None,
                 secondary_model_path=None,
                 cattle_model_path=None,
                 device='cuda:0'
                 ):
        self.device_config = _device_config
        self.color_format = color_format
        self.color_resolution = color_resolution
        self.depth_mode = depth_mode
        self.k4a_dll_path = r"C:\Program Files\Azure Kinect SDK v1.4.2\sdk\windows-desktop\amd64\release\bin\k4a.dll"
        self.primary_model_path = primary_model_path
        self.secondary_model_path = secondary_model_path
        self.cattle_model_path = cattle_model_path
        self.device = device

        # 截图保存位置
        save_dir = "./captured_data"
        self.save_dir = save_dir
        self.count=0    # 命名ID
        #创建保存目录
        os.makedirs(os.path.join(save_dir,"color"),exist_ok=True)
        os.makedirs(os.path.join(save_dir,"depth"),exist_ok=True)
        os.makedirs(os.path.join(save_dir,"pointcloud"),exist_ok=True)

        # 目标检测器初始化
        self.deployer = None
        if not (primary_model_path and secondary_model_path):
            raise ValueError("必须指定primary_model_path和secondary_model_path")
        self.deployer = TwoStageDeployer(
            primary_model_path=primary_model_path,
            secondary_model_path=secondary_model_path,
            cattle_model_path=cattle_model_path,
            conf_udder=0.4,
            conf_hoof=0.25,
            conf_secondary=0.3,
            device='cuda:0',
            max_udder=1,
            max_hoof=4,
            max_nipple=4,
        )

    # 保存截图
    def save_data(self,capture,color_image,depth_image):
        # 保存彩色图
        cv2.imwrite(f"{self.save_dir}/color/{self.count}.jpg",color_image)
        # 保存深度数据
        np.save(f"{self.save_dir}/depth/{self.count}.npy",depth_image)
        # 保存原始点云
        ret_points,points = capture.get_transformed_pointcloud()
        if ret_points:
            np.save(f"{self.save_dir}/pointcloud/{self.count}.npy",points)

    # 窗口可视化
    def update_visualization(self,img):
        # cv2接受的图像格式是BGR，需要将RGB图像转为BGR格式
        cv2.imshow('RGB Image',img)

    # 运行
    def _run(self):

        # 初始化配置
        pykinect.initialize_libraries(module_k4a_path=self.k4a_dll_path)
        device = pykinect.start_device(config=self.device_config)
        # 初始化显示RGB图像窗口
        cv2.namedWindow('RGB Image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RGB Image',640,480)


        try:
            while True:

                # 获取图像
                capture = device.update()
                # 获取彩色图像
                ret_color , bgra32_image = capture.get_color_image()
                # 获取深度数据
                ret_depth , depth_image = capture.get_depth_image()

                # 验证结果
                if not ret_color or not ret_depth:
                    continue

                # 停止 截图
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_data(capture,bgra32_image,depth_image)
                    self.count+=1


                # 更新窗口
                self.update_visualization(bgra32_image)
        finally:
            device.close()
            cv2.destroyAllWindows()






if __name__ == "__main__":
    PRIMARY_MODEL_PATH = r"D:\yolov8-main\ultralytics\runs\detect\train_hoof_hard_finetune\weights\best.pt"
    SECONDARY_MODEL_PATH = r"D:\yolov8-main\ultralytics\runs\detect\train_nipple_finetune\weights\best.pt"
    CATTLE_MODEL_PATH = r"D:\yolov8-main\ultralytics\runs\detect\train_cattle\weights\best.pt"

    run = Run(
        _device_config= pykinect.default_configuration,
        color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32,
        color_resolution=pykinect.K4A_COLOR_RESOLUTION_1080P,
        depth_mode=pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED,

        primary_model_path = PRIMARY_MODEL_PATH,
        secondary_model_path = SECONDARY_MODEL_PATH,
        cattle_model_path=CATTLE_MODEL_PATH,
    )
    run._run()