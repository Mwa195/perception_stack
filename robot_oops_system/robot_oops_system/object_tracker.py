import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from my_interfaces_pkg.msg import TrackedObject, TrackedObjects
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTrackingNode(Node):
    """
    Object Tracking Node: receives the segmentation mask from topic
    /segmentation_mask and traces highlighted areas in the mask then
    draw boxes around each detected object in the main feed processed
    from /camera_feed topic and then publishes the id, class and coordinates
    of each object
    """
    def __init__(self):
        """
        Initialize the object_tracking_node
        """
        super().__init__("object_tracking_node")

        # DeepSORT tracker
        self.tracker = DeepSort(max_age=30)

        # OpenCV Bridge
        self.bridge = CvBridge()

        # Store the latest segmentation mask
        self.latest_mask = None

        # Define the color map and corresponding class labels (same as semantic seg nod)
        self.class_map = {
            (0, 0, 0): "Background",
            (128, 0, 0): "Aeroplane",
            (0, 128, 0): "Bicycle",
            (128, 128, 0): "Bird",
            (0, 0, 128): "Boat",
            (128, 0, 128): "Bottle",
            (0, 128, 128): "Bus",
            (128, 128, 128): "Car",
            (64, 0, 0): "Cat",
            (192, 0, 0): "Chair",
            (64, 128, 0): "Cow",
            (192, 128, 0): "Dining Table",
            (64, 0, 128): "Dog",
            (192, 0, 128): "Horse",
            (64, 128, 128): "Motorbike",
            (192, 128, 128): "Person",
            (0, 64, 0): "Potted Plant",
            (128, 64, 0): "Sheep",
            (0, 192, 0): "Sofa",
            (128, 192, 0): "Train",
            (0, 64, 128): "Monitor"
        }

        # Convert color map to numpy array for fast comparison
        self.class_colors = np.array(list(self.class_map.keys()))
        self.class_labels = list(self.class_map.values())

        self.tracked_ids = {}  # Stores tracked Ids
        self.available_ids = []  # Priority queue for available Ids
        self.next_id = 1  # Next ID to assign

        # Subscribers
        self.feed_sub = self.create_subscription(
            Image,
            "/camera_feed",
            self._feed_callback,
            10
            )
        
        self.mask_sub = self.create_subscription(
            Image,
            "/segmentation_mask",
            self._mask_callback,
            10
            )

        # Publisher
        self.tracked_pub = self.create_publisher(
            TrackedObjects,
            "/tracked_objects",
            10
            )
        
        self.get_logger().info("Object Tracking Node started ✅")

    def _mask_callback(self, msg: Image):
        """
        Stores the segmentation mask received from /segmentation_mask topic
        """
        self.latest_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _extract_objects_from_mask(self):
        """
        Finds objects in the segmentation mask based on predefined colors
        """
        if self.latest_mask is None:
            self.get_logger().warn("No segmentation mask received yet")
            return []

        detected_objects = []

        for i, color in enumerate(self.class_colors):
            class_label = self.class_labels[i]

            if class_label == "Background":
                continue  # Ignore background

            # Create a binary mask for this class
            mask = cv2.inRange(self.latest_mask, color, color)

            # Find contours which detects object boundaries
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                if area > 20000:  # Ignore small noise
                    detected_objects.append(([x, y, x + w, y + h], 1.0, class_label))

        return detected_objects

    def _feed_callback(self, msg: Image):
        """
        Processes frames, extracts objects from segmentation mask and tracks
        them when receiving the feed
        """
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Get objects from the segmentation mask
        detections = self._extract_objects_from_mask()

        self.get_logger().info(f"📌 Detected Objects: {detections}")

        if not detections:
            self.get_logger().warn("No valid detections to track")
            return

        # Update object tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)

        # Get currently tracked IDs
        current_tracked_ids = set(track.track_id for track in tracks if track.is_confirmed())

        # Release IDs for objects that are no longer in the frame
        for track_id in list(self.tracked_ids.keys()):
            if track_id not in current_tracked_ids:
                self.available_ids.append(self.tracked_ids.pop(track_id))  # Reuse this ID

        # Sort available IDs to always reuse the lowest first
        self.available_ids.sort()

        # Prepare tracked objects message
        tracked_objects_msg = TrackedObjects()
        tracked_objects_msg.objects = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_label = track.det_class

            # Assign a stable ID
            if track_id not in self.tracked_ids:
                if self.available_ids:
                    self.tracked_ids[track_id] = self.available_ids.pop(0)  # Reuse lowest available ID
                else:
                    self.tracked_ids[track_id] = self.next_id
                    self.next_id += 1  # Increase ID counter if no reusable ID is available

            # Set the current tracked Id 
            new_id = self.tracked_ids[track_id]

            self.get_logger().info(f"📦 Tracking ID {new_id}: {ltrb} -> {class_label}")

            x_min, y_min, x_max, y_max = map(int, ltrb)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame.shape[1] - 1, x_max)
            y_max = min(frame.shape[0] - 1, y_max)

            # Ignore small detections (noise filtering)
            min_area = 20000
            if (x_max - x_min) * (y_max - y_min) < min_area:
                continue

            # Get the color for this class
            bbox_color = tuple(self.class_colors[self.class_labels.index(class_label)])
            bbox_color = tuple(map(int, bbox_color))  # Ensure it's a valid BGR tuple

            # Draw bounding box with the correct class color
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), bbox_color, 3)
            cv2.putText(frame, f'ID: {new_id} {class_label}', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, bbox_color, 3)

            # Create TrackedObject msg
            tracked_obj = TrackedObject()
            tracked_obj.id = int(new_id)
            tracked_obj.class_label = class_label
            tracked_obj.xmin = float(x_min)
            tracked_obj.ymin = float(y_min)
            tracked_obj.xmax = float(x_max)
            tracked_obj.ymax = float(y_max)
            # Add to the list
            tracked_objects_msg.objects.append(tracked_obj)
        # Publish
        self.tracked_pub.publish(tracked_objects_msg)

        cv2.imshow("tracked_output", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()