import rclpy
from rclpy.node import Node
from my_interfaces_pkg.msg import TrackedObjects, ObjectsSpeeds, ObjectData, ObjectsData

class FusionNode(Node):
    """
    Data Dusion Node: collects data about each object from topics 
    /tracked_objects and /treacked_speeds and matches data for each
    then publishes a full list to /objects_list
    """
    def __init__(self):
        """
        Initialize the data_fusion_node
        """
        super().__init__("data_fusion_node")

        self.timer = self.create_timer(1/30, self._objects_list_publisher)
        self.objects_from_tracker = []
        self.objects_from_optical_flow = []

        # Subscribers
        self.objects_sub = self.create_subscription(
            TrackedObjects,
            "/tracked_objects",
            self._objects_callback,
            10
        )

        self.speeds_sub = self.create_subscription(
            ObjectsSpeeds,
            "/tracked_speeds",
            self._speeds_callback,
            10
        )

        # Publisher
        self.list_pub = self.create_publisher(
            ObjectsData,
            "/objects_list",
            10
        )

        # Publish
        self.get_logger().info("Data Fusion Node Started ✅")

    def _objects_callback(self, msg: TrackedObjects):
        """
        Receives list of objects from tracker node
        """
        self.objects_from_tracker = msg.objects
        
    def _speeds_callback(self, msg: ObjectsSpeeds):
        """
        Receives list of objects' speeds from optical flow node
        """
        self.objects_from_optical_flow = msg.objects

    def _objects_list_publisher(self):
        """
        main function which fuses data and publishes the final full list
        """
        if not self.objects_from_tracker or not self.objects_from_optical_flow:
            return
        
        objects_list = ObjectsData()

        # Match Objects from both lists
        for obj in self.objects_from_tracker:
            for speed in self.objects_from_optical_flow:
                if obj.id == speed.id:
                    obj_data = ObjectData()
                    obj_data.id = speed.id
                    obj_data.class_label = obj.class_label
                    obj_data.speed_kmh = speed.speed_kmh
                    obj_data.xmin = obj.xmin
                    obj_data.ymin = obj.ymin
                    obj_data.xmax = obj.xmax
                    obj_data.ymax = obj.ymax

                    objects_list.objects.append(obj_data)
        # Publish
        self.list_pub.publish(objects_list)


def main(args=None):
    rclpy.init(args=args) # Initialize ROS2
    node = FusionNode() # Create Node as an instance of the Class
    try:
        rclpy.spin(node) # Keep the node spinning
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown() # Shutdown the node gracefully

if __name__ == "__main__":
    main()