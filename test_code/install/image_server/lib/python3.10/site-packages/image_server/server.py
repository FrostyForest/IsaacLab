import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class ImagePublisher(Node):

    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, 'image_topic', 10)
        self.br = CvBridge()
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def publish_numpy_image(self, numpy_image):
        # ... (代码同上) ...
           # 检查 NumPy 数组的数据类型
        if numpy_image.dtype != np.uint8:
            raise ValueError("NumPy 数组必须是 uint8 类型")

        # 检查 NumPy 数组的维度
        if len(numpy_image.shape) != 2 and len(numpy_image.shape) != 3:
            raise ValueError("NumPy 数组必须是 2 维 (灰度) 或 3 维 (彩色)")

        # 确定图像编码
        if len(numpy_image.shape) == 2:
            encoding = "mono8"
        elif numpy_image.shape[2] == 3:
            encoding = "bgr8"
        elif numpy_image.shape[2] == 4:
            encoding = "bgra8"
        else:
            raise ValueError("不支持的彩色通道数量")

        # 使用 cv_bridge 将 NumPy 数组转换为 ROS 2 Image 消息
        image_msg = self.br.cv2_to_imgmsg(numpy_image, encoding=encoding)

        # 设置消息头信息 (可选)
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = "camera_frame" # 根据实际情况修改

        # 发布消息
        self.publisher_.publish(image_msg)

    def timer_callback(self):
        # ... (代码同上) ...
        # 生成或获取 NumPy 格式的图像数据
        numpy_image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8) # 示例
        # 发布图像
        self.publish_numpy_image(numpy_image)

def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    rclpy.spin(image_publisher)
    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()