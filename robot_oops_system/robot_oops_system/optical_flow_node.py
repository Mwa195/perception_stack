import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from my_interfaces_pkg.msg import TrackedObjects, ObjectSpeed, ObjectsSpeeds

class OpticalFlowNode(Node):
    """
    Optical Flow Node: receives the data of the tracked objects from
    /tracked_objects topic to identify ROIs and processes the speed of
    each object from the main feed received from /camera_feed then publishes
    speed data to /tracked_speeds
    """
    def __init__(self):
        """
        Initialize the optical_flow_node
        """
        super().__init__('optical_flow_node')

        self.bridge = CvBridge()
        self.prev_gray = None
        self.tracked_features = {}  # Stores previous points per object ID

        # Shi-Tomasi parameters
        self.feature_params = dict(maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7)

        # Lucas-Kanade optical flow params
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.fps = 30
        self.latest_objects = []

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera_feed',
            self._image_callback,
            10
            )
        
        self.objects_sub = self.create_subscription(
            TrackedObjects,
            '/tracked_objects',
            self._objects_callback,
            10
            )

        # Publisher
        self.speed_pub = self.create_publisher(
            ObjectsSpeeds,
            '/tracked_speeds',
            10
            )
        
        self.get_logger().info("Optical Flow Node Started ✅")

    def _objects_callback(self, msg: TrackedObjects):
        """
        Receives tracked objects from the Object tracker node
        """
        self.latest_objects = msg.objects  # Store latest objects

    def _image_callback(self, msg: Image):
        """
        Processes optical flow for each object and publishes speed
        """
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return

        tracked_speeds_msg = ObjectsSpeeds()
        tracked_speeds_msg.objects = []

        for obj in self.latest_objects:
            obj_id = obj.id
            x_min, y_min, x_max, y_max = int(obj.xmin), int(obj.ymin), int(obj.xmax), int(obj.ymax)

            # Get feature points in the object region
            mask = np.zeros_like(gray)
            mask[y_min:y_max, x_min:x_max] = 255
            new_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params)

            if new_pts is None:
                self.tracked_features.pop(obj_id, None)  # Remove if no features are found
                continue  # Skip if no points found

            # Retrieve previous points
            prev_pts = self.tracked_features.get(obj_id, new_pts)

            # Compute optical flow
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts, None, **self.lk_params)

            if new_pts is not None and status is not None:
                good_new = new_pts[status == 1]
                good_old = prev_pts[status == 1]

                speeds = []
                for new, old in zip(good_new, good_old):
                    distance = np.linalg.norm(new - old)
                    speed_kmh = (distance * self.fps * 3.6) / 100
                    speeds.append(speed_kmh)

                avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

                # Store new points for the next frame
                self.tracked_features[obj_id] = new_pts

                # Create ObjectData message
                object_data = ObjectSpeed()
                object_data.id = obj_id
                object_data.speed_kmh = avg_speed

                tracked_speeds_msg.objects.append(object_data)

                self.get_logger().info(f"📦 Object {obj_id}: Speed = {avg_speed:.2f} km/h")

        # Update previous frame
        self.prev_gray = gray.copy()
        
        # Publish
        self.speed_pub.publish(tracked_speeds_msg)

def main(args=None):
    rclpy.init(args=args) # Initialize ROS2
    node = OpticalFlowNode() # Create Node as an instance of the Class
    try:
        rclpy.spin(node) # Keep the node spinning
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown() # Shutdown the node gracefully

if __name__ == '__main__':
    main()
