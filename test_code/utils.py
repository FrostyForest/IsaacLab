import torch
import cv2

def show_tensor_rgba_image(tensor_image):
    """
    将 PyTorch Tensor 格式的 RGBA 图像转换为 OpenCV 格式并显示。

    Args:
        tensor_image: PyTorch Tensor 格式的 RGBA 图像，形状为 (1, H, W, 4)。
    """

    # 检查 Tensor 形状
    if tensor_image.shape != (1, 480, 640, 4):
        raise ValueError("输入 Tensor 的形状必须是 (1, 480, 640, 4).")

    # 将 Tensor 转换为 NumPy 数组，并调整通道顺序
    image = tensor_image.permute(1, 2, 0).cpu().numpy()

    # 将图像数据类型转换为 uint8
    image = (image * 255).astype("uint8")

    # 将 RGBA 转换为 BGRA (OpenCV 使用 BGRA 格式)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

    # 显示图像
    cv2.imshow("Tensor RGBA Image", image)
    cv2.waitKey(0)  # 按下任意键关闭窗口
    cv2.destroyAllWindows()
