import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from my_interfaces_pkg.msg import TrackedObject, TrackedObjects
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTrackingNode(Node):
    def __init__(self):
        super().__init__("object_tracking_node")

        # Subscribers
        self.feed_sub = self.create_subscription(Image, "/camera_feed", self.feed_callback, 10)
        self.mask_sub = self.create_subscription(Image, "/segmentation_mask", self.mask_callback, 10)

        # Publisher
        self.tracked_pub = self.create_publisher(TrackedObjects, "/tracked_objects", 10)

        # DeepSORT tracker
        self.tracker = DeepSort(max_age=30)

        # OpenCV Bridge
        self.bridge = CvBridge()

        # Store the latest segmentation mask
        self.latest_mask = None

        # Define the color map and corresponding class labels
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

        # Convert color map to NumPy array for fast comparison
        self.class_colors = np.array(list(self.class_map.keys()))
        self.class_labels = list(self.class_map.values())

        self.get_logger().info("✅ Object Tracking Node started.")

    def mask_callback(self, msg: Image):
        """Stores the latest segmentation mask."""
        self.latest_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def extract_objects_from_mask(self):
        """Finds objects in the segmentation mask based on predefined colors."""
        if self.latest_mask is None:
            self.get_logger().warn("⚠️ No segmentation mask received yet.")
            return []

        detected_objects = []

        for i, color in enumerate(self.class_colors):
            class_label = self.class_labels[i]

            if class_label == "Background":
                continue  # Ignore background

            # Create a binary mask for this class
            mask = cv2.inRange(self.latest_mask, color, color)

            # Find contours (object boundaries)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                if area > 20000:  # Ignore small noise
                    detected_objects.append(([x, y, x + w, y + h], 1.0, class_label))

        return detected_objects

    def feed_callback(self, msg: Image):
        """Processes frames, extracts objects from segmentation mask, and tracks them."""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Get objects from the segmentation mask
        detections = self.extract_objects_from_mask()

        self.get_logger().info(f"📌 Detected Objects: {detections}")

        if not detections:
            self.get_logger().warn("⚠️ No valid detections to track.")
            return

        # Update object tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)

        # Prepare tracked objects message
        tracked_objects_msg = TrackedObjects()
        tracked_objects_msg.objects = []

        # ID remapping for sequential numbering
        id_map = {}  # Dictionary to store remapped IDs
        next_id = 1  # Start numbering from 1

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_label = track.det_class

            # Assign new sequential ID if it's not already assigned
            if track_id not in id_map:
                id_map[track_id] = next_id
                next_id += 1  # Increment for the next object

            new_id = id_map[track_id]

            self.get_logger().info(f"📦 Tracking ID {new_id}: {ltrb} -> {class_label}")

            x_min, y_min, x_max, y_max = map(int, ltrb)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame.shape[1] - 1, x_max)
            y_max = min(frame.shape[0] - 1, y_max)

            # Ignore small detections (noise filtering)
            min_area = 20000  # Adjust as needed
            if (x_max - x_min) * (y_max - y_min) < min_area:
                continue  # Skip small detections

            # Get the correct color for this class
            bbox_color = tuple(self.class_colors[self.class_labels.index(class_label)])
            bbox_color = tuple(map(int, bbox_color))  # Ensure it's a valid BGR tuple

            # Draw bounding box with the correct class color
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), bbox_color, 3)
            cv2.putText(frame, f'ID: {new_id} {class_label}', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, bbox_color, 3)

            # Create TrackedObject message
            tracked_obj = TrackedObject()
            tracked_obj.id = int(new_id)
            tracked_obj.class_label = class_label
            tracked_obj.xmin = float(x_min)
            tracked_obj.ymin = float(y_min)
            tracked_obj.xmax = float(x_max)
            tracked_obj.ymax = float(y_max)

            tracked_objects_msg.objects.append(tracked_obj)

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


# import rclpy
# from rclpy.node import Node
# import cv2
# import numpy as np
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# from my_interfaces_pkg.msg import TrackedObject, TrackedObjects
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort

# class ObjectTrackingNode(Node):
#     def __init__(self):
#         super().__init__("object_tracking_node")

#         # Subscribers
#         self.feed_sub = self.create_subscription(Image, "/camera_feed", self.feed_callback, 10)
#         self.mask_sub = self.create_subscription(Image, "/segmentation_mask", self.mask_callback, 10)

#         # Publisher
#         self.tracked_pub = self.create_publisher(TrackedObjects, "/tracked_objects", 10)

#         # YOLO model for object detection
#         self.model = YOLO("yolov8n.pt")

#         # DeepSORT tracker
#         self.tracker = DeepSort(max_age=30)

#         # OpenCV Bridge
#         self.bridge = CvBridge()

#         # Store the latest segmentation mask
#         self.latest_mask = None

#         # Allowed class names for detection (all lowercase)
#         self.allowed_class_names = {"person", "car", "bus", "bottle", "chair"}
#         # Display mapping for neat labels
#         self.display_map = {
#             "person": "Person",
#             "car": "Car",
#             "bus": "Bus",
#             "bottle": "Bottle",
#             "chair": "Chair"
#         }

#         self.get_logger().info("✅ Object Tracking Node started.")

#     def mask_callback(self, msg: Image):
#         """Stores the latest segmentation mask."""
#         mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
#         # Convert mask to grayscale so that pixel values represent class indices
#         self.latest_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

#     def filter_mask_by_class(self):
#         """Filters the segmentation mask based on allowed classes.
#            For this example, we assume that the segmentation model outputs 135 for 'person'."""
#         if self.latest_mask is None:
#             self.get_logger().warn("⚠️ No segmentation mask received yet.")
#             return None

#         detected_classes = np.unique(self.latest_mask)
#         self.get_logger().info(f"🟢 Segmentation Mask Classes: {detected_classes}")

#         # Create a binary mask: assume segmentation outputs 135 for person
#         filtered_mask = np.zeros_like(self.latest_mask, dtype=np.uint8)
#         filtered_mask[self.latest_mask == 135] = 255

#         return filtered_mask

#     def feed_callback(self, msg: Image):
#         """Processes frames, applies segmentation mask, detects objects, and tracks them."""
#         frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

#         # Ensure segmentation mask exists and filter it
#         filtered_mask = self.filter_mask_by_class()
#         if filtered_mask is None:
#             self.get_logger().warn("⚠️ Skipping frame - No valid mask.")
#             return

#         # Apply the mask to the frame
#         masked_frame = cv2.bitwise_and(frame, frame, mask=filtered_mask)
#         cv2.imshow("Masked Frame for YOLO", masked_frame)
#         cv2.waitKey(1)

#         # Run YOLO detection on the masked frame
#         results = self.model(masked_frame)

#         detections = []
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 confidence = float(box.conf[0])
#                 class_id = int(box.cls[0])
#                 # Get detection name in lowercase for filtering
#                 class_name = self.model.names[class_id].lower()
#                 self.get_logger().info(f"🎯 YOLO DETECTION: {class_name} at {(x1, y1, x2, y2)} (Conf: {confidence})")
#                 # Filter detections by allowed class names
#                 if class_name in self.allowed_class_names:
#                     detections.append(([x1, y1, x2, y2], confidence, class_name))

#         self.get_logger().info(f"📌 Detected Objects: {detections}")

#         if not detections:
#             self.get_logger().warn("⚠️ No valid detections to track.")
#             return

#         # Update object tracker (using full frame for tracking visualization)
#         tracks = self.tracker.update_tracks(detections, frame=frame)

#         tracked_objects_msg = TrackedObjects()
#         tracked_objects_msg.objects = []

#         for track in tracks:
#             if not track.is_confirmed():
#                 continue

#             track_id = track.track_id
#             ltrb = track.to_ltrb()
#             # Use display mapping to get a nicely formatted label
#             display_label = self.display_map.get(track.det_class, "Unknown")
#             self.get_logger().info(f"📦 Tracking ID {track_id}: {ltrb} -> {display_label}")

#             # Clamp bounding box coordinates within frame dimensions
#             x_min, y_min, x_max, y_max = map(int, ltrb)
#             x_min = max(0, x_min)
#             y_min = max(0, y_min)
#             x_max = min(frame.shape[1] - 1, x_max)
#             y_max = min(frame.shape[0] - 1, y_max)

#             # Draw bounding box and label on the frame
#             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)  # RED BOX
#             cv2.putText(frame, f'ID: {track_id} {display_label}', (x_min, y_min - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)  # WHITE TEXT

#         #     # Create TrackedObject message
#         #     tracked_obj = TrackedObject()
#         #     tracked_obj.id = track_id
#         #     tracked_obj.class_label = display_label
#         #     tracked_obj.xmin = float(x_min)
#         #     tracked_obj.ymin = float(y_min)
#         #     tracked_obj.xmax = float(x_max)
#         #     tracked_obj.ymax = float(y_max)

#         #     tracked_objects_msg.objects.append(tracked_obj)

#         # self.tracked_pub.publish(tracked_objects_msg)

#         cv2.imshow("tracked_output", frame)
#         cv2.waitKey(1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = ObjectTrackingNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         rclpy.shutdown()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
