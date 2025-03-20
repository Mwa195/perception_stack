import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraStreamer(Node):
    """
    Camera Streamer Node: captures frames from the laptop's webcam
    and publishes it to /camera_feed topic
    """
    def __init__(self):
        """
        Initialize the camera_streamer node
        """
        super().__init__('camera_streamer')

        # Open Camera (ID: 0)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Camera didn't Open")
            return
        
        # CV Bridge (Used to convert the frames into ROS2 msg *Image*)
        self.bridge = CvBridge()

        # Main Timer which captures the frames with 30 FPS
        self.main_timer = self.create_timer(1/30, self._stream_video)

        # Publisher
        self.feed_pub = self.create_publisher(
            Image,
            '/camera_feed',
            10
            )
        
        self.get_logger().info("Camera Streamer Node has started ✅")

    def _stream_video(self):
        """
        Continuously capture and display frames from the webcam
        """
        # Read Frames from teh camera
        ret, frame = self.cap.read()
        if ret:
            cv2.imshow("Camera Feed", frame)
            # COnvert frame to Image format
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.feed_pub.publish(ros_image) # Publish to the topic
        else:
            self.get_logger().warn("Failed to capture frame")
        
        # Prees e to exit
        if cv2.waitKey(1) & 0xFF == ord('e'):
            self.get_logger().info("Shutting down camera stream...")
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()
        

def main(args=None):
    rclpy.init(args=args) # Initialize ROS2
    node = CameraStreamer() # Create Node as an instance of the Class
    try:
        rclpy.spin(node) # Keep the node spinning
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown() # Shutdown the node gracefully

if __name__ == '__main__':
    main()

