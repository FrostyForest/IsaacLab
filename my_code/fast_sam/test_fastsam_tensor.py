import cv2
import torch
import numpy as np
import torch.nn.functional as F
from fastsam import FastSAM, FastSAMPrompt
from pathlib import Path


# ... letterbox_tensor 函数定义保持不变 ...
def letterbox_tensor(
    im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32
):
    # (此处省略函数代码，与之前相同)
    was_3d = False
    if im.dim() == 3:
        was_3d = True
        im = im.unsqueeze(0)
    shape = im.shape[2:]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = F.interpolate(im, size=new_unpad, mode="bilinear", align_corners=False)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    pad_color_value = color[0] / 255.0 if im.is_floating_point() else color[0]
    im = F.pad(im, (left, right, top, bottom), "constant", value=pad_color_value)
    if was_3d:
        im = im.squeeze(0)
    return im, r, (dw, dh)


# --- 1. 初始化模型 ---
model = FastSAM("my_code/fast_sam/FastSAM-x.pt")  # 确保路径正确
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- 2. 从单张图片创建 Tensor ---
image_path = "/home/linhai/图片/截图/截图 2025-07-01 21-31-53.png"  # 确保路径正确
orig_img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)

# 将 NumPy 图像转换为 (C, H, W) 的 Tensor，并直接移动到 GPU
source_tensor_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).to(device)

# --- 预处理单张图片的 Tensor ---
imgsz = 640
conf = 0.8
iou = 0.95

input_tensor_float = source_tensor_rgb.float()
tensor_padded, _, _ = letterbox_tensor(input_tensor_float, new_shape=imgsz, stride=32, auto=True)
tensor_normalized_single = tensor_padded / 255.0  # (C, H, W)

# --- 3. 使用 expand 模拟批处理 ---
batch_size = 3
# 添加 batch 维度 (1, C, H, W) 然后 expand 到 (N, C, H, W)
batch_tensor = tensor_normalized_single.unsqueeze(0).expand(batch_size, -1, -1, -1)

# --- START OF FIX: 为 `orig_imgs` 参数准备一个匹配长度的列表 ---
# 因为批处理中的每个元素都源自同一张图片，我们创建一个包含N个相同 NumPy 图像的列表
list_of_orig_imgs_for_postprocess = [orig_img_bgr] * batch_size
# --- END OF FIX ---


# --- 4. 使用模型进行预测 ---
print(f"Input batch tensor shape (from expand): {batch_tensor.shape}")
print(f"Length of orig_imgs list: {len(list_of_orig_imgs_for_postprocess)}")

# 调用 predict_tensor 方法
everything_results = model.predict_tensor(
    batch_tensor,
    orig_imgs=list_of_orig_imgs_for_postprocess,  # 传入我们准备好的列表
    retina_masks=False,
    conf=conf,
    iou=iou,
)

# --- 5. 循环处理批处理结果 ---
OUTPUT_DIR = "output/expand_batch_results"
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

if everything_results and len(everything_results) == batch_size:
    print("Prediction successful, processing expanded batch results...")

    # 循环遍历批次中的每一个结果
    for i, single_result in enumerate(everything_results):
        print(f"\n--- Processing item {i+1}/{batch_size} from expanded batch ---")

        # 获取与当前结果对应的原始 NumPy 图像 (它们都是一样的)
        current_orig_img_bgr = list_of_orig_imgs_for_postprocess[i]

        # 为当前这张图片和它的结果创建一个 Prompt 处理器
        prompt_process = FastSAMPrompt(current_orig_img_bgr, [single_result], device=device)

        num_objects = len(single_result)
        if num_objects > 0:
            masks = single_result.masks.data
            print(f"    检测到 {len(masks)} 个对象。")

            # a) 保存带有所有掩码标注的原始图像
            annotated_img_path = Path(OUTPUT_DIR) / f"annotated_image_{i+1}.jpg"
            prompt_process.plot(
                annotations=masks,
                output_path=str(annotated_img_path),
            )
            print(f"    Annotated image saved to: {annotated_img_path}")

            # b) 将每个掩码保存到该图片专属的子文件夹中
            masks_output_dir = Path(OUTPUT_DIR) / f"image_{i+1}_masks"
            masks_output_dir.mkdir(exist_ok=True)
            for j, mask in enumerate(masks):
                mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                mask_filename = masks_output_dir / f"mask_{j+1}.png"
                cv2.imwrite(str(mask_filename), mask_np)
            print(f"    Individual masks saved to: {masks_output_dir}")

        else:
            print("    No objects found in this image.")
else:
    print(
        f"Prediction failed or returned incorrect number of results. Expected {batch_size}, got {len(everything_results) if everything_results else 0}."
    )
