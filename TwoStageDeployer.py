from ultralytics import YOLO
import cv2
import os
import numpy as np

class TwoStageDeployer:
    def __init__(self,
                 primary_model_path,
                 secondary_model_path,
                 cattle_model_path,
                 conf_secondary=0.25,
                 conf_udder=0.4,
                 conf_hoof=0.25,
                 conf_cattle = 0.8,
                 max_udder=1,
                 max_hoof=6,
                 max_nipple=4,
                 max_cattle=1,
                 roi_expand=0.3,
                 device = 'cuda:0'):
        self.model_primary = YOLO(primary_model_path).to(device)
        self.model_secondary = YOLO(secondary_model_path).to(device)
        self.model_cattle = YOLO(cattle_model_path).to(device)
        self.conf_secondary = conf_secondary
        self.conf_udder = conf_udder
        self.conf_hoof = conf_hoof
        self.conf_cattle = conf_cattle
        self.max_udder = max_udder
        self.max_hoof = max_hoof
        self.max_nipple = max_nipple
        self.max_cattle = max_cattle
        self.roi_expand = roi_expand
        self.device = device
        self.class_names = {0:"udder",1:"hoof",2:"nipple"}
        self.output_dir = "two_stage_results"
        os.makedirs(self.output_dir, exist_ok=True)

    def expand_roi(self,img,bbox):
        """
        扩展检测框区域，返回扩展后的ROI和偏移量
        :param img: 输入图像
        :param bbox: 原始检测框 [x1, y1, x2, y2]
        :return: (扩展后的ROI图像, (x偏移量, y偏移量))
        """
        try:
            h,w = img.shape[:2]
            x1,y1,x2,y2 = map(int, bbox[:4])

            # 计算扩展量（基于检测框大小的比例扩展）
            w_extend = (x2-x1)*self.roi_expand
            h_extend = (y2-y1)*self.roi_expand

            # 计算扩展后的边界（避免越界）
            x1_exp = max(0,x1-int(w_extend))
            y1_exp = max(0,y1-int(h_extend))
            x2_exp = min(w,x2+int(w_extend))
            y2_exp = min(h,y2+int(h_extend))

            # 处理ROI区域无效的情况（避免空图像）
            if x1_exp>=x2_exp or y1_exp>=y2_exp:
                print(f"ROI区域无效（x1_exp={x1_exp},x2_exp={x2_exp},y1_exp={y1_exp},y2_exp={y2_exp}）.使用原始框")
                x1_exp,y1_exp,x2_exp,y2_exp = x1, y1, x2, y2

            # 提取扩展后的ROI
            roi_img = img[y1_exp:y2_exp,x1_exp:x2_exp]
            offset = (x1_exp,y1_exp)

            # 校验偏移量格式
            if len(offset) !=2:
                raise ValueError(f"偏移量格式错误，预期2个值，实际{len(offset)}个：{offset}")
            if not all(isinstance(v,int) for v in offset):
                raise TypeError(f"偏移量类型错误，预期整数，实际：{offset}")

            return roi_img,offset
        except Exception as e:
            print(f"ROI扩展失败：{str(e)}")
            # 返回原始框作为 fallback
            x1,y1,x2,y2 = map(int, bbox[:4])
            return img[y1:y2,x1:x2],(x1,y1)

    def filter_by_max_count(self,boxes,max_count,cls_name):
        """按置信度排序，保留前max_count个目标，过滤多余检测"""
        if len(boxes)<=max_count:
            return boxes

        # 按置信度降序排序（boxes格式：[x1,y1,x2,y2,conf,cls_id]）
        boxes_sorted = sorted(boxes, key=lambda x:x[4],reverse=True)
        filtered = boxes_sorted[:max_count]
        print(f"过滤{cls_name}：原{len(boxes)}个 → 保留{max_count}个（置信度最高）")
        return filtered

    def draw_boxes(self,img,udder_boxes,hoof_boxes,nipple_boxes):
        """在图像上绘制检测框"""
        # 绘制乳房（绿色）
        for bbox in udder_boxes:
            # 校验检测框长度（预期6个值：x1,y1,x2,y2,conf,cls_id）
            if len(bbox) < 6:
                print(f"警告：乳房检测框格式错误，预期6个值，实际{len(bbox)}个，跳过绘制")
                continue
            x1, y1, x2, y2, conf, _ = bbox
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"udder({conf:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 绘制蹄部（蓝色）
        for bbox in hoof_boxes:
            if len(bbox) < 6:
                print(f"警告：蹄部检测框格式错误，预期6个值，实际{len(bbox)}个，跳过绘制")
                continue
            x1, y1, x2, y2, conf, _ = bbox
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(img, f"hoof({conf:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 绘制乳头（红色）
        for bbox in nipple_boxes:
            if len(bbox) < 6:
                print(f"警告：乳头检测框格式错误，预期6个值，实际{len(bbox)}个，跳过绘制")
                continue
            x1, y1, x2, y2, conf, _ = bbox
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img, f"nipple({conf:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def process_rgb_images(self,rgb_images):
        """直接处理内存中的RGB图像（cv2格式），返回检测框"""
        try:
            rgb_copy = rgb_images.copy()
            h,w = rgb_images.shape[:2]

            # 第一阶段检测
            try:
                result_primary = self.model_primary(
                    rgb_images,conf=0.1,device=self.device,verbose=False
                )[0]
            except Exception as e:
                print(f"第一阶段模型推理失败：{str(e)}")
                return {'annotated_image': rgb_copy, 'udder': [], 'hoof': [], 'nipple': []}

            udder_boxes = []
            hoof_boxes = []
            for box in result_primary.boxes:
                try:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    # 强制确保bbox是4个值（x1,y1,x2,y2）
                    if len(bbox)!=4:
                        print(f"警告：模型返回的检测框格式错误，预期4个值，实际{len(bbox)}个，跳过")
                        continue

                    # 组装6个值的检测框（x1,y1,x2,y2,conf,cls_id）
                    if cls_id==0 and conf>=self.conf_udder:
                        udder_boxes.append(bbox+[conf,cls_id])
                    elif cls_id==1 and conf>=self.conf_hoof:
                        hoof_boxes.append(bbox+[conf,cls_id])
                except Exception as e:
                    print(f"处理第一阶段检测框失败：{str(e)}")
                    continue

            # 应用数量约束
            udder_boxes = self.filter_by_max_count(udder_boxes, self.max_udder, "udder")
            hoof_boxes = self.filter_by_max_count(hoof_boxes, self.max_hoof, "hoof")

            # 第二阶段乳头检测
            nipple_boxes = []
            for udder in udder_boxes:
                try:
                    if len(udder)<4:
                        print(f"警告：乳房检测框格式错误，无法提取ROI，跳过乳头检测")
                        continue
                    udder_bbox = udder[:4]

                    roi,offset = self.expand_roi(rgb_images,udder_bbox)
                    if len(offset)!=2:
                        print(f"偏移量解包失败，预期2个值，实际{len(offset)}个")
                        continue
                    x_offset,y_offset = offset

                    # 乳头模型推理
                    result_secondary = self.model_secondary(roi,conf=self.conf_secondary,device=self.device,verbose=False)[0]
                    for box in result_secondary.boxes:
                        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        # 组装6个值的乳头检测框
                        nipple_boxes.append([x1+x_offset,y1+y_offset,x2+x_offset,y2+y_offset,conf,2])
                except Exception as e:
                    print(f"处理乳头检测失败：{str(e)}")
                    continue

            nipple_boxes = self.filter_by_max_count(nipple_boxes,self.max_nipple,"nipple")

            # 绘制检测框
            self.draw_boxes(rgb_copy,udder_boxes, hoof_boxes, nipple_boxes)

            return {
                "annotated_image": rgb_copy,
                "udder": udder_boxes,
                "hoof": hoof_boxes,
                "nipple":nipple_boxes
            }
        except Exception as e:
            print(f"推理失败: {str(e)}")
            return {
                'annotated_image': rgb_images,  # 返回原始图像
                'udder': [],
                'hoof': [],
                'nipple': []
            }

    def process_image_old(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"图像不存在: {img_path}")
        img_copy = img.copy()

        # 第一阶段：检测乳房和蹄部
        results_primary = self.model_primary(
            img, conf=0.1, device=self.device, verbose=False    # 基础阈值设0.1，避免提前过滤目标
        )[0]

        # 提取乳房和蹄部，带置信度
        udder_boxes = []  # 格式：[x1,y1,x2,y2,conf,cls_id]
        hoof_boxes = []
        for box in results_primary.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().tolist()
            # 乳房：用高阈值self.conf_udder过滤（如0.4）
            if cls_id == 0 and conf >=self.conf_udder:
                udder_boxes.append(bbox + [conf, cls_id])
            elif cls_id == 1 and conf >=self.conf_hoof:
                hoof_boxes.append(bbox + [conf, cls_id])

        # 应用数量约束：过滤多余乳房和蹄部
        udder_boxes = self.filter_by_max_count(udder_boxes, self.max_udder, "乳房")
        hoof_boxes = self.filter_by_max_count(hoof_boxes, self.max_hoof, "蹄部")

        # 第二阶段：乳头检测（仅在有效乳房区域内）
        nipple_boxes = []
        for udder in udder_boxes:
            # 提取乳房边界框（前4个元素是x1,y1,x2,y2）
            udder_bbox = udder[:4]
            roi, (x_offset, y_offset) = self.expand_roi(img, udder_bbox)
            results_secondary = self.model_secondary(
                roi, conf=self.conf_secondary, device=self.device, verbose=False
            )[0]
            # 映射乳头坐标到原图，并记录置信度
            for box in results_secondary.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                nipple_boxes.append([
                    x1 + x_offset, y1 + y_offset,
                    x2 + x_offset, y2 + y_offset,
                    conf, 2  # cls_id=2
                ])

        # 应用数量约束：过滤多余乳头
        nipple_boxes = self.filter_by_max_count(nipple_boxes, self.max_nipple, "乳头")

        # 标注所有目标
        # 1. 乳房（绿色）
        for bbox in udder_boxes:
            x1, y1, x2, y2, conf, cls_id = bbox
            cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img_copy, f"udder({conf:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 2. 蹄部（蓝色）
        for bbox in hoof_boxes:
            x1, y1, x2, y2, conf, cls_id = bbox
            cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img_copy, f"hoof({conf:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 3. 乳头（红色）
        for bbox in nipple_boxes:
            x1, y1, x2, y2, conf, cls_id = bbox
            cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(img_copy, f"nipple({conf:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 保存结果
        save_path = os.path.join(self.output_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, img_copy)
        print(f"结果保存至: {save_path}")

        return {
            "udder": udder_boxes,
            "hoof": hoof_boxes,
            "nipple": nipple_boxes
        }

    # 检测牛整体
    def full_detect(self,img):
        """
        检测牛整体
        :param img: 接收图像
        :return: {接收的图像，牛整体检测框}
        """
        result_cattle = self.model_cattle(
            img,conf = self.conf_cattle,device = self.device,verbose=False
        )[0]

        cattle_boxes = []
        for box in result_cattle.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy.cpu().numpy().tolist()[0]
            cattle_boxes.append(bbox+[conf,cls_id])
        return {
            "annotated_image":img,
            "cattle_boxes":cattle_boxes
        }

    # 检测牛部位
    def part_detect(self,img):
        """
        检测牛部位
        :param img: 接收图像
        :return {接收的图像，牛乳房检测框，牛肢蹄检测框，牛乳头检测框}
        """
        result_primary = self.model_primary(
            img,conf = 0.1,device = self.device,verbose=False
        )[0]

        udder_boxes = []
        hoof_boxes = []
        for box in result_primary.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().tolist()
            if cls_id ==0 and conf>=self.conf_udder:
                udder_boxes.append(bbox+[conf,cls_id])
            elif cls_id ==1 and conf>=self.conf_hoof:
                hoof_boxes.append(bbox+[conf,cls_id])

        udder_boxes = self.filter_by_max_count(udder_boxes,self.max_udder,"udder")
        hoof_boxes = self.filter_by_max_count(hoof_boxes,self.max_hoof,"hoof")

        nipple_boxes = []
        for udder in udder_boxes:
            udder_bbox = udder[:4]
            roi,(x_offset,y_offset) = self.expand_roi(img,udder_bbox)
            results_secondary = self.model_secondary(
                roi,conf = self.conf_secondary,device = self.device,verbose=False
            )[0]

            for box in results_secondary.boxes:
                x1,y1,x2,y2= box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                nipple_boxes.append([
                    x1+x_offset , y1+y_offset,
                    x2+x_offset , y2+y_offset,
                    conf,2
                ])

        nipple_boxes = self.filter_by_max_count(nipple_boxes,self.max_nipple,"nipple")

        return {
            "annotated_image":img,
            "udder_boxes":udder_boxes,
            "hoof_boxes":hoof_boxes,
            "nipple_boxes":nipple_boxes
        }

    # 处理牛整体
    def full_process(self,full_detect_result):
        """
        标注牛整体检测框（紫色）
        :param full_detect_result: 接收牛整体检测结果
        :return 带上检测框后的图像
        """
        img_copy = full_detect_result["annotated_image"]
        for bbox in full_detect_result["cattle_boxes"]:
            x1,y1,x2,y2,conf,cls_id = bbox
            cv2.rectangle(img_copy,(int(x1),int(y1)),(int(x2),int(y2)),(128,0,128),2)
            cv2.putText(img_copy,f"cattle({conf:.2f})",(int(x1),int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,0,128), 2)

        return img_copy

    # 处理牛部位
    def part_process(self,img,part_detect_result):
        """
        标注牛部位检测框
        :param img 接收的图像
        :param part_detect_result 牛部位检测结果
        :return 带上检测框后的图像
        """
        img_copy = img.copy()

        # 乳房（绿色）
        for bbox in part_detect_result["udder_boxes"]:
            x1 ,y1,x2,y2,conf,cls_id = bbox
            cv2.rectangle(img_copy,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            cv2.putText(img_copy,f"udder({conf:.2f})",(int(x1),int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # 肢蹄（蓝色）
        for bbox in part_detect_result["hoof_boxes"]:
            x1 ,y1,x2,y2,conf,cls_id = bbox
            cv2.rectangle(img_copy,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
            cv2.putText(img_copy,f"hoof({conf:.2f})",(int(x1),int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        # 乳头（红色）
        for bbox in part_detect_result["nipple_boxes"]:
            x1,y1,x2,y2,conf,cls_id = bbox
            cv2.rectangle(img_copy,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
            cv2.putText(img_copy,f"nipple({conf:.2f})",(int(x1),int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        return img_copy

    # 整合全部处理，返回带检测框的图片
    def process_image(self,image_path):
        """
        总处理
        :param image_path: 图像路径
        :return 处理后的图像
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Not found {image_path}")
        img_copy = img.copy()

        full_detect_result = self.full_detect(img)  # 检测牛整体
        part_detect_result = self.part_detect(img)  # 检测牛部位
        img_copy = self.full_process(full_detect_result)    # 处理牛整体
        img_copy = self.part_process(img_copy,part_detect_result)   # 处理牛部位

        # 保存结果
        save_path = os.path.join(self.output_dir,os.path.basename(image_path))
        cv2.imwrite(save_path,img_copy)
        print(f"save {save_path}")

        return img_copy

if __name__ == "__main__":
    deployer = TwoStageDeployer(
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
    # deployer.process_image(r"D:\yolov8-main\datasets\Resources\cattle_images\images\train\15097S.JPG")
    deployer.process_image(r"E:\Project Code\Python\YOD-PCP-KIADC\captured_data\color\0.jpg")
    # deployer.process_image_old(r"D:\yolov8-main\datasets\Resources\cattle_images\images\train\15097S.JPG")