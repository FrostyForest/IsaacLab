import torch
import numpy as np
import cv2
from pathlib import Path
from fastsam import FastSAM, FastSAMPrompt
import time


def generate_all_masks_in_batch(
    model_path, image_paths: list, output_dir, device="cuda", imgsz=1024, conf=0.4, iou=0.9
):
    """
    高效地为一批（多张）图像生成所有对象的掩码。

    Args:
        model_path (str): FastSAM模型的路径。
        image_paths (list): 包含多个输入图像路径的列表。
        output_dir (str): 保存结果的根目录。
        device (str): 计算设备 ('cuda' or 'cpu')。
        imgsz (int): 推理的图像尺寸。
        conf (float): 对象检测的置信度阈值。
        iou (float): NMS的IoU阈值。
    """
    # --- 1. 设置和加载模型 (一次性) ---
    print("开始批处理...")
    print(f"使用设备: {device}, 处理 {len(image_paths)} 张图片。")

    # 加载FastSAM模型
    print(f"正在加载模型: {model_path}")
    model = FastSAM(model_path)

    # --- 2. 执行批处理推理 (核心步骤) ---
    # 将整个列表传递给模型，这是实现高效率的关键
    print("正在执行批处理推理...")
    everything_results = model(
        image_paths,  # 传入路径列表
        device=device,
        retina_masks=True,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
    )
    print("推理完成，正在保存所有结果...")

    # --- 3. 遍历每张图片的结果并保存 ---
    for i, single_image_results in enumerate(everything_results):

        # 获取原始图片路径和基本名称
        original_image_path = image_paths[i]
        base_name = Path(original_image_path).stem  # e.g., "dogs" from "images/dogs.jpg"

        print(f"  -> 正在保存第 {i+1}/{len(image_paths)} 张图片的结果: {base_name}")

        # 为当前图片创建一个独立的输出子目录
        image_output_dir = Path(output_dir) / base_name
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # 使用FastSAMPrompt来辅助处理单张图片的结果
        prompt_process = FastSAMPrompt(original_image_path, single_image_results, device=device)

        # 检查是否检测到任何对象
        try:
            masks = prompt_process.results[0].masks.data
        except (AttributeError, IndexError):
            print(f"    在 {base_name} 中未检测到任何对象。")
            continue

        num_masks = len(masks)
        print(f"    检测到 {num_masks} 个对象。")

        # a) 保存带有所有掩码标注的原始图像
        annotated_img_path = image_output_dir / "annotated_image.jpg"
        prompt_process.plot(
            annotations=prompt_process.everything_prompt(),
            output_path=str(annotated_img_path),
        )

        # b) 将每个掩码保存为单独的二值图像
        masks_output_dir = image_output_dir / "individual_masks"
        masks_output_dir.mkdir(exist_ok=True)

        for j, mask in enumerate(masks):
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            mask_filename = masks_output_dir / f"mask_{j+1}.png"
            cv2.imwrite(str(mask_filename), mask_np)

    print(f"\n所有 {len(image_paths)} 张图片的处理和保存均已完成！")
    print(f"结果保存在根目录: {output_dir}")


if __name__ == "__main__":
    # --- 配置参数 ---
    MODEL_PATH = "my_code/fast_sam/FastSAM-x.pt"

    # TODO: 将此列表更改为你的多个图片路径
    IMAGE_PATHS = [
        "/home/linhai/图片/截图/截图 2025-07-01 21-31-53.png",
        "/home/linhai/图片/appl.jpeg",
        # 你可以在这里添加更多图片路径
        # "/path/to/your/image3.png",
        # "/path/to/your/image4.jpeg",
    ]

    # 设置保存结果的根目录
    OUTPUT_DIR = "output/batch_results"

    # 自动选择设备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    t1 = time.time()
    generate_all_masks_in_batch(
        model_path=MODEL_PATH,
        image_paths=IMAGE_PATHS,
        output_dir=OUTPUT_DIR,
        device=DEVICE,
        imgsz=320,
        conf=0.6,
        iou=0.9,
    )
    t2 = time.time()
    print("time", t2 - t1)
