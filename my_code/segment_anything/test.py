# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import torch

import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torchvision.io import read_image
from torchvision.transforms import functional as F
import time

# --- 1. 设置 ---
# 输入图片路径
IMAGE_PATH = "/home/linhai/图片/截图/截图 2025-07-01 21-31-53.png"
# 输出文件夹路径
OUTPUT_DIR = "output_objects"
# SAM模型权重路径
SAM_CHECKPOINT = "/home/linhai/code/IsaacLab/my_code/segment_anything/sam_vit_b_01ec64.pth"  # 使用最小的ViT-B模型
MODEL_TYPE = "vit_b"

# 检查输出目录是否存在，如果不存在则创建
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用设备: {device}")

# --- 2. 加载模型和生成器 ---
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)

# SamAutomaticMaskGenerator可以调整很多参数来优化分割效果
# 详见官方文档: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py
mask_generator = SamAutomaticMaskGenerator(sam, pred_iou_thresh=0.99, points_per_batch=64)

# --- 3. 读取图像并生成蒙版 ---
print("正在读取图像...")
# 使用OpenCV读取图像，注意OpenCV读取的是BGR格式
image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    print(f"错误: 无法读取图像 {IMAGE_PATH}")
    exit()

# 将图像从BGR转换为RGB，因为SAM需要RGB格式
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

while 1:
    print("正在生成所有对象的蒙版...")
    t1 = time.time()
    # generate会返回一个列表，每个元素是一个字典，包含一个分割对象的信息
    masks = mask_generator.generate(image_rgb)
    print(f"完成！共找到 {len(masks)} 个对象。")
    t1_1 = time.time()
    print("mask time", t1_1 - t1)
    # --- 4. 遍历蒙版，抠图并保存 ---
    print("正在处理并保存每个对象...")
    # 按面积从大到小排序，可以优先处理主要物体
    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)

    for i, mask_data in enumerate(sorted_masks):
        # 获取二进制蒙版 (True/False)
        mask = mask_data["segmentation"]
        area = mask_data["area"]
        predicted_iou = mask_data["predicted_iou"]
        print(area, predicted_iou)

        # 创建一个4通道的透明背景图像 (RGBA)
        # 尺寸与原图相同
        output_image = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)

        # 将原图的RGB通道复制到新图像
        output_image[:, :, :3] = image_rgb

        # 使用蒙版来设置Alpha通道
        # 在物体区域，Alpha为255（不透明），其他区域为0（透明）
        output_image[:, :, 3] = mask * 255

        # --- 裁剪图像以减少空白区域 ---
        # 获取边界框信息 [x, y, width, height]
        x, y, w, h = mask_data["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        # 根据边界框裁剪图像
        cropped_image = output_image[y : y + h, x : x + w, :]

        # # --- 保存裁剪后的图像 ---
        # # 定义输出文件名
        # filename = os.path.join(OUTPUT_DIR, f"object_{i+1}.png")

        # # OpenCV在写入时需要BGRA格式，所以需要从RGBA转换
        # cropped_image_bgra = cv2.cvtColor(cropped_image, cv2.COLOR_RGBA2BGRA)

        # # 保存为PNG文件以保留透明度
        # cv2.imwrite(filename, cropped_image_bgra)
    t2 = time.time()
    print("cost time", t2 - t1)
    print(f"所有对象已成功保存到 '{OUTPUT_DIR}' 文件夹中。")
