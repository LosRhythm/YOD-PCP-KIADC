import open3d as o3d
import numpy as np
import cv2
from TwoStageDeployer import TwoStageDeployer



class PointCloudProcessor:
    def __init__(self,voxel_size_full = 0.02,voxel_size_part = 0.005):
                 self.voxel_size_full = voxel_size_full
                 self.voxel_size_part = voxel_size_part
                 self.full_pcd = o3d.geometry.PointCloud()
                 self.part_pcds = {"hoof":[],"udder":[],"nipple":[]}

                 self.detector = TwoStageDeployer(
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


    def crop_pcd_from_box(self,pcd,color_img,box):
        """根据2D目标框裁剪点云（需映射到3D坐标）"""
        h,w = color_img.shape[:2]
        x1 ,x2 ,y1 ,y2 = map(int,box)
        # 点云索引对应图像坐标（简化：假设点云与彩色图像素一一对应）
        indices = []
        for i in range(len(pcd.points)):
            # 实际需通过内参将3D点投影到2D，这里简化为像素坐标范围筛选
            u = int(i%w)
            v = int(i//w)
            if x1<=u<=x2 and y1<=v<=y2:
                indices.append(i)
        return pcd.select_by_index(indices)


    def process_single_view(self, color_img, pcd_path, detector):
        """处理单视角数据：检测→裁剪→下采样"""
        # 1. 检测整体牛
        full_cattle = self.detector.full_detect(color_img)
        cattle_boxes = full_cattle["cattle_boxes"]
        if len(cattle_boxes) == 0:
            print("No cattle boxes detected,skip this view")
            return

        # 2. 加载点云并裁剪出牛的整体
        pcd = o3d.io.read_point_cloud(pcd_path)
        cow_pcd = self.crop_pcd_from_box(pcd,color_img,cattle_boxes[0]) # 取第一个牛的框
        cow_pcd_down = cow_pcd.voxel_down_sample(voxel_size=self.voxel_size_full)
        self.full_pcd += cow_pcd_down

        # 3. 检测局部特征并裁剪
        parts_cattle = self.detector.part_detect(color_img)
        udder_boxes = parts_cattle["udder_boxes"]
        hoof_boxes = parts_cattle["hoof_boxes"]
        nipple_boxes = parts_cattle["nipple_boxes"]

        for box in udder_boxes:
            part_pcd = self.crop_pcd_from_box(pcd,color_img,box)
            part_pcd_down = part_pcd.voxel_down_sample(voxel_size=self.voxel_size_part)
            self.part_pcds["hoof"].append(part_pcd_down)

        for box in hoof_boxes:
            part_pcd = self.crop_pcd_from_box(pcd,color_img,box)
            part_pcd_down = part_pcd.voxel_down_sample(voxel_size=self.voxel_size_part)
            self.part_pcds["hoof"].append(part_pcd_down)

        for box in udder_boxes:
            part_pcd = self.crop_pcd_from_box(pcd,color_img,box)
            part_pcd_down = part_pcd.voxel_down_sample(voxel_size=self.voxel_size_part)
            self.part_pcds["hoof"].append(part_pcd_down)

    def register_multi_view(self):
        """配准多视角点云（拼接为完整模型）"""
        # 简化：使用ICP算法迭代配准（实际可能需要特征点匹配初始化）
        if len(self.full_pcd.points) <100:
            return

        # 假设已有一个初始点云，后续点云与之配准
        target = self.full_pcd
        for i in range(1,len(self.full_pcd.points)):    # 示例逻辑，需根据实际数据调整
            source = self.full_pcd
            # ICP配准
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source=source,target=target,max_correspondence_distance=self.voxel_size_full*2,init=np.eye(4),
                method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            source.transform(reg_p2p.transformation)
            target+=source
        self.full_pcd = target

    def calculate_metrics(self):
        """计算局部特征的指标"""
        metrics = {}
        # 1. 肢蹄：计算平均尺寸
        if self.part_pcds["hoof"]:
            hoof_sizes = []
            for pcd in self.part_pcds["hoof"]:
                bbox = pcd.get_axis_aligned_bounding_box()
                size = bbox.get_extent()  # [x, y, z]尺寸
                hoof_sizes.append(size)
            metrics["hoof_avg_size"] = np.mean(hoof_sizes, axis=0)

        # 2. 乳房：计算体积（通过凸包）
        if self.part_pcds["udder"]:
            udder_pcd = o3d.geometry.PointCloud()
            for p in self.part_pcds["udder"]:
                udder_pcd += p
            hull = o3d.geometry.TriangleMesh.create_from_point_cloud_convex_hull(udder_pcd)
            metrics["udder_volume"] = hull.get_volume()

        # 3. 乳头：数量和间距
        if self.part_pcds["nipple"]:
            nipple_pcd = o3d.geometry.PointCloud()
            for p in self.part_pcds["nipple"]:
                nipple_pcd += p
            metrics["nipple_count"] = len(nipple_pcd.points)
            # 计算平均间距
            if len(nipple_pcd.points) >= 2:
                dists = o3d.geometry.PointCloud.compute_point_cloud_distance(nipple_pcd, nipple_pcd)
                metrics["nipple_avg_distance"] = np.mean(dists)

        return metrics

