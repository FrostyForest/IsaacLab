
import numpy as np
import rclpy
from rclpy.node import Node
from image_server.image_server.server import  ImagePublisher
import time

rclpy.init()
publisher = ImagePublisher()

while 1:
    numpy_image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
    publisher.publish_numpy_image(numpy_image)
    time.sleep(1)
    print('1')