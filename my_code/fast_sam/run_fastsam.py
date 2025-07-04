import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from fastsam import FastSAM, FastSAMPrompt
import time


def generate_all_masks_efficiently(model_path, image_path, output_dir, device="cuda", imgsz=320, conf=0.7, iou=0.9):
    """
    高效地为单个图像生成所有对象的掩码。

    Args:
        model_path (str): FastSAM模型的路径 (例如 'FastSAM-x.pt')。
        image_path (str): 输入图像的路径。
        output_dir (str): 保存结果的目录。
        device (str): 计算设备 ('cuda' or 'cpu')。
        imgsz (int): 推理的图像尺寸。减小此值可大幅提升速度。
        conf (float): 对象检测的置信度阈值。
        iou (float): NMS（非极大值抑制）的IoU阈值。
    """
    # --- 1. 设置和加载模型 ---
    print("开始处理...")
    print(f"使用设备: {device}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载FastSAM模型
    print(f"正在加载模型: {model_path}")
    model = FastSAM(model_path)

    # --- 2. 执行推理 ---
    # retina_masks=True可以获得更高质量的分割结果，但速度会稍慢。
    # 为了极致的速度，可以设置为 retina_masks=False
    print(f"正在处理图像: {image_path}")
    t1 = time.time()
    everything_results = model(
        image_path,
        device=device,
        retina_masks=False,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
    )
    t2 = time.time()
    print(t2 - t1)

    # --- 3. 处理和保存结果 ---
    # 使用FastSAMPrompt来辅助处理结果
    prompt_process = FastSAMPrompt(image_path, everything_results, device=device)

    # 检查是否检测到任何对象
    try:
        masks = prompt_process.results[0].masks.data
    except (AttributeError, IndexError):
        print("未在图像中检测到任何对象。")
        return

    num_masks = len(masks)
    print(f"检测到 {num_masks} 个对象。")

    # --- 4. 保存结果 ---
    # a) 保存带有所有掩码标注的原始图像
    annotated_img_path = output_path / "annotated_image.jpg"
    prompt_process.plot(
        annotations=prompt_process.everything_prompt(),
        output_path=str(annotated_img_path),
    )
    print(f"已将标注图像保存至: {annotated_img_path}")

    # b) 将每个掩码保存为单独的二值图像
    masks_output_dir = output_path / "individual_masks"
    masks_output_dir.mkdir(exist_ok=True)

    original_image = cv2.imread(image_path)

    for i, mask in enumerate(masks):
        # 将mask张量转换为numpy数组，并调整为与原图相同的尺寸
        mask_np = mask.cpu().numpy().astype(np.uint8)

        # 创建一个纯黑色的背景
        binary_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.uint8)
        # 将掩码区域设置为白色 (255)
        binary_mask[mask_np > 0] = 255

        # 保存二值掩码图像
        mask_filename = masks_output_dir / f"mask_{i+1}.png"
        cv2.imwrite(str(mask_filename), binary_mask)

    print(f"已将 {num_masks} 个单独的掩码图像保存至: {masks_output_dir}")
    print("处理完成！")


if __name__ == "__main__":
    # --- 配置参数 ---
    # TODO: 确保你已经下载了模型权重文件
    MODEL_PATH = "my_code/fast_sam/FastSAM-x.pt"  # 或者使用更快的 'FastSAM-s.pt'

    # TODO: 将此路径更改为你的图片路径
    IMAGE_PATH = "/home/linhai/图片/截图/截图 2025-07-01 21-31-53.png"

    # 设置保存结果的目录
    OUTPUT_DIR = "output/masks"

    # 自动选择设备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    generate_all_masks_efficiently(
        model_path=MODEL_PATH,
        image_path=IMAGE_PATH,
        output_dir=OUTPUT_DIR,
        device=DEVICE,
        imgsz=320,  # 尝试减小到 640 以获得更快的速度
        conf=0.8,
        iou=0.95,
    )
