import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraStreamer(Node):
    def __init__(self):
        super().__init__('camera_streamer')

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Camera didn't Open")
            return
        
        self.feed_pub = self.create_publisher(Image, '/camera_feed', 10)
        self.bridge = CvBridge()

        self.main_timer = self.create_timer(1/30, self.stream_video)
        
        self.get_logger().info("Camera Streamer Node has started.")

    def stream_video(self):
        """
        Continuously captures and displays frames from the webcam
        """
        ret, frame = self.cap.read()
        if ret:
            cv2.imshow("Camera Feed", frame)
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.feed_pub.publish(ros_image)
        else:
            self.get_logger().warn("Failed to capture frame")
        

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('e'):
            self.get_logger().info("Shutting down camera stream...")
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    node = CameraStreamer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()

