#! /home/mwa/opencv_env/bin/python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import torchvision.models as models
from torchvision import transforms

class SegmentationNode(Node):
    """
    Semantic Segmentation Node which subscribes to the /camera_feed topic
    and performs semantic segmentation on the feed to identify objects
    according to VOC 20 then publish the result to /segmentation_mask
    """
    def __init__(self):
        """
        Initialize the segmentation_node
        """
        super().__init__("segmentation_node")

        # Subscriber
        self.feed_sub = self.create_subscription(
            Image,
            "/camera_feed",
            self.feed_callback,
            10
            )
        
        # Publisher
        self.mask_pub = self.create_publisher(
            Image,
            "/segmentation_mask",
            10
            )
        
        # CV Bridge (Used to convert the frames into ROS2 msg *Image*)
        self.bridge = CvBridge()

        # Load the pre-trained DeepLabV3 model ResNet101
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.eval()

        # Move the model to GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Define the preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ])

        self.get_logger().info("DeepLabV3+ Segmentation Node started.")

    def feed_callback(self, msg: Image):
        """
        Feed callback function which is executed whenever the node receives a
        msg from the /camera_feed topic in order to process the frame and
        create the segmentation mask
        """
        # Convert ROS2 Image msg to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Preprocess the image
        input_tensor = self.preprocess(frame)  # Convert to tensor and normalize
        input_batch = input_tensor.unsqueeze(0).to(self.device)  # Add batch dimension and move to device

        # Perform inference
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)  # Get the predicted class for each pixel

        # Convert the output to a numpy array
        mask = output_predictions.byte().cpu().numpy()

        # Apply a color map to the mask
        mask_colored = self.apply_color_map(mask)

        # Display the results
        cv2.imshow("DeepLabV3+ Segmentation", mask_colored)
        # COnvert frame to Image format
        mask_msg = self.bridge.cv2_to_imgmsg(mask_colored, encoding="bgr8")
        # Publish the results
        self.mask_pub.publish(mask_msg)
        cv2.waitKey(1)  # Required for OpenCV to update the window

    def apply_color_map(self, mask):
        """
        Apply a color map to the segmentation mask in order to differentiate
        between detected classes
        """
        # Pascal VOC color map (21 classes)
        colors = [
            [0, 0, 0],       # Background
            [128, 0, 0],     # Aeroplane
            [0, 128, 0],     # Bicycle
            [128, 128, 0],   # Bird
            [0, 0, 128],     # Boat
            [128, 0, 128],   # Bottle
            [0, 128, 128],   # Bus
            [128, 128, 128], # Car
            [64, 0, 0],      # Cat
            [192, 0, 0],     # Chair
            [64, 128, 0],    # Cow
            [192, 128, 0],   # Dining Table
            [64, 0, 128],    # Dog
            [192, 0, 128],   # Horse
            [64, 128, 128],  # Motorbike
            [192, 128, 128], # Person
            [0, 64, 0],      # Potted Plant
            [128, 64, 0],    # Sheep
            [0, 192, 0],     # Sofa
            [128, 192, 0],   # Train
            [0, 64, 128]     # TV/Monitor
        ]
        classes_names = [
            "Background",       # Background
            "Aeroplane",     # Aeroplane
            "Bicycle",     # Bicycle
            "Bird",   # Bird
            "Boat",     # Boat
            "Bottle",   # Bottle
            "Bus",   # Bus
            "Car", # Car
            "Cat",      # Cat
            "Chair",     # Chair
            "Cow",    # Cow
            "Dining Table",   # Dining Table
            "Dog",    # Dog
            "Horse",   # Horse
            "Motorbike",  # Motorbike
            "Person", # Person
            "Potted Plant",      # Potted Plant
            "Sheep",    # Sheep
            "Sofa",     # Sofa
            "Train",   # Train
            "Monitor"     # TV/Monitor
        ]
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for i in range(21):  # Loop over all classes
            colored_mask[mask == i] = colors[i]
        
        # # Log detected classes
        # detected_classes = set(mask.flatten()) 
        # for i in detected_classes:
        #     if i > 0:  # Ignore background (0)
        #         self.get_logger().info(f"Detected {classes_names[i]}/s")
                
        return colored_mask

def main(args=None):
    rclpy.init(args=args) # Initialize ROS2
    node = SegmentationNode() # Create Node as an instance of the Class
    try:
        rclpy.spin(node) # Keep the node spinning
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown() # Shutdown the node gracefully

if __name__ == "__main__":
    main()