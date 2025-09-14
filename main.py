import cv2
import numpy as np
import open3d as o3d
from ImageCapture import ImageCapture
from PointCloudProcessor import PointCloudProcessor
from TwoStageDeployer import TwoStageDeployer
import pykinect_azure as pykinect
PRIMARY_MODEL_PATH = r"D:\yolov8-main\ultralytics\runs\detect\train_hoof_hard_finetune\weights\best.pt"
SECONDARY_MODEL_PATH = r"D:\yolov8-main\ultralytics\runs\detect\train_nipple_finetune\weights\best.pt"
CATTLE_MODEL_PATH = r"D:\yolov8-main\ultralytics\runs\detect\train_cattle\weights\best.pt"

def main():
    # 1. 采集多视角数据
    capture = ImageCapture(
        _device_config=pykinect.default_configuration,
        color_format=pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32,
        color_resolution=pykinect.K4A_COLOR_RESOLUTION_1080P,
        depth_mode=pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED,

        primary_model_path=PRIMARY_MODEL_PATH,
        secondary_model_path=SECONDARY_MODEL_PATH,
        cattle_model_path=CATTLE_MODEL_PATH,
    )
    print("Start capturing (press 's' to save, 'q' to exit after capture)")
    capture.run()

    # 2. 初始化检测器和点云处理器
    detector = TwoStageDeployer(
        primary_model_path=r"D:\yolov8-main\ultralytics\runs\detect\train_hoof_hard_finetune\weights\best.pt",
        # 微调后的乳房肢蹄模型
        secondary_model_path=r"D:\yolov8-main\ultralytics\runs\detect\train_nipple_finetune\weights\best.pt",
        # 微调后的乳头模型
        cattle_model_path=r"D:\yolov8-main\ultralytics\runs\detect\train_cattle\weights\best.pt",
        conf_udder=0.4,  # 乳房阈值：0.4（无假阳性，符合其100%精确率）
        conf_hoof=0.5,  # 肢蹄阈值：0.25（漏检率低，符合其95.3%召回率）
        conf_secondary=0.3,  # 乳头阈值：0.3（减少误检）
        max_udder=1,
        max_hoof=4,
        max_nipple=4,
        device='cuda:0'
    )
    processor = PointCloudProcessor()

    # 3. 处理每个视角的数据
    import os
    data_dir = "./captured_data"
    for i in range(capture.count):
        color_img = cv2.imread(f"{data_dir}/color/{i}.jpg")
        pcd_path = f"{data_dir}/pointcloud/{i}.npy"
        print(f"Processing view {i}...")
        processor.process_single_view(color_img, pcd_path, detector)

    # 4. 配准多视角点云，生成完整模型
    processor.register_multi_view()
    o3d.io.write_point_cloud("full_cow_model.npy", processor.full_pcd)

    # 5. 计算指标
    metrics = processor.calculate_metrics()
    print("Calculated metrics:", metrics)

if __name__ == "__main__":
    main()