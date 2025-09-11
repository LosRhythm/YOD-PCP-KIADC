import cv2

import pykinect_azure as pykinect
from TwoStageDeployer import TwoStageDeployer


class Run:
    def __init__(self,
                 _device_config,
                 color_format,
                 color_resolution,
                 depth_mode,
                 primary_model_path=None,
                 secondary_model_path=None,
                 device='cuda:0'
                 ):
        self.device_config = _device_config
        self.color_format = color_format
        self.color_resolution = color_resolution
        self.depth_mode = depth_mode
        self.k4a_dll_path = r"C:\Program Files\Azure Kinect SDK v1.4.2\sdk\windows-desktop\amd64\release\bin\k4a.dll"
        self.primary_model_path = primary_model_path
        self.secondary_model_path = secondary_model_path
        self.device = device


        # 目标检测器初始化
        self.deployer = None
        if not (primary_model_path and secondary_model_path):
            raise ValueError("必须指定primary_model_path和secondary_model_path")
        self.deployer = TwoStageDeployer(
            primary_model_path=primary_model_path,
            secondary_model_path=secondary_model_path,
            conf_udder=0.4,
            conf_hoof=0.25,
            conf_secondary=0.3,
            device='cuda:0',
            max_udder=1,
            max_hoof=4,
            max_nipple=4,
        )

    def initialize(self):
        pykinect.initialize_libraries(module_k4a_path=self.k4a_dll_path)
        device = pykinect.start_device(config=self.device_config)

        try:
            while True:
                capture = device.update()
                # 获取彩色图像
                ret , bgra32_image = capture.get_color_image()

                if not ret:
                    continue

                # 将BGRA32转为RGB图像
                bgr_image = bgra32_image[:, :, :3]
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

                # 检测目标
                try:
                    detection_result = self.deployer.process_rgb_images(rgb_image)
                except Exception as e:
                    print(f"ERROR:{str(e)}")
                    return

                # 检查检测结果完整性
                # required_keys = ['annotated_img', 'udder', 'hoof', 'nipple']
                # if not all(key in detection_result for key in required_keys):
                #     print(f"检测结果格式错误，缺少必要键：{detection_result.keys()}")
                #     return

                annotated_img = detection_result['annotated_image']
                all_boxes = {
                    'udder':detection_result['udder'],
                    'hoof':detection_result['hoof'],
                    'nipple':detection_result['nipple']
                }

                # 更新窗口
                self.update_visualization(annotated_img)


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            device.close()
            cv2.destroyAllWindows()

    # 窗口可视化
    def update_visualization(self,annotated_img):
        # 显示标注后的RGB图像
        cv2.namedWindow('RGB Image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RGB Image',640,480)
        # cv2接受的图像格式是BGR，需要将RGB图像转为BGR格式
        cv2.imshow('RGB Image',cv2.cvtColor(annotated_img,cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    PRIMARY_MODEL_PATH = r"D:\yolov8-main\ultralytics\runs\detect\train_hoof_hard_finetune\weights\best.pt"
    SECONDARY_MODEL_PATH = r"D:\yolov8-main\ultralytics\runs\detect\train_nipple_finetune\weights\best.pt"

    run = Run(
        _device_config= pykinect.default_configuration,
        color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32,
        color_resolution=pykinect.K4A_COLOR_RESOLUTION_720P,
        depth_mode=pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED,

        primary_model_path = PRIMARY_MODEL_PATH,
        secondary_model_path = SECONDARY_MODEL_PATH,
    )
    run.initialize()