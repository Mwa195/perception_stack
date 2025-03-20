# perception_stack
# **Object Tracking and Speed Estimation in ROS2**

## **📌 Overview**
This ROS2-based project performs **real-time object tracking and speed estimation** using:
- **Semantic Segmentation** to detect objects in a scene.
- **DeepSORT Tracking** to assign IDs and track objects across frames.
- **Optical Flow (Lucas-Kanade)** to estimate object speeds.
- **Data Fusion** to combine tracked object positions and speed estimates.
- **Video Streaming** to publish real-time camera feed.

The system uses **ROS2 Jazzy** and subscribes to image feeds to detect, track, and estimate the speed of multiple moving objects.

---

## **📂 Project Structure**
```
ros2_ws/
│-- src/
│   ├── robot_oops_system/
│   │   ├── nodes/
│   │   │   ├── data_fusion.py
│   │   │   ├── object_tracker.py
│   │   │   ├── optical_flow_node.py
│   │   │   ├── streamer.py
│   │   │   ├── semantic_segmentation_dl.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   ├── setup.cfg
│   ├── my_interfaces_pkg/
│   │   ├── msg/
│   │   │   ├── TrackedObject.msg
│   │   │   ├── TrackedObjects.msg
│   │   │   ├── ObjectData.msg
│   │   │   ├── ObjectsData.msg
│   │   │   ├── ObjectSpeed.msg
│   │   │   ├── ObjectsSpeeds.msg
│   │   ├── CMakeLists.txt
│   │   ├── package.xml
│-- README.md  # Project Documentation
```

---

## **🚀 Setup Instructions**
### **1️⃣ Install Dependencies**
```bash
sudo apt update && sudo apt install -y \
    python3-opencv \
    ros-jazzy-cv-bridge \
    ros-jazzy-rclpy \
    ros-jazzy-sensor-msgs \
    python3-pip
pip install torch torchvision torchaudio deepsort-realtime
```
### **2️⃣ Clone the Repository**
```bash
cd ~/ros2_ws/src  # Navigate to your workspace
git clone https://github.com/Mwa195/perception_stack.git
cd .. && colcon build --symlink-install
source install/setup.bash
```

### **3️⃣ Run the System**
Start each node in separate terminals:
```bash
# Start the Video Streaming Node
ros2 run robot_oops_system streamer_node

# Start the Segmentation Node
ros2 run robot_oops_system semantic_segmentation_dl

# Start the Object Tracking Node
ros2 run robot_oops_system object_tracker

# Start the Optical Flow Node
ros2 run robot_oops_system optical_flow_node

# Start the Data Fusion Node
ros2 run robot_oops_system data_fusion
```

---

## **🛠 Nodes & Functionality**

### **📡 Video Streaming Node** (`streamer.py`)
- **Publishes to:**
  - `/camera_feed` (real-time video frames)
- **Key Features:**
  - Captures live video from a connected camera.
  - Publishes frames as ROS2 Image messages.

### **🎨 Segmentation Node** (`semantic_segmentation_dl.py`)
- **Subscribes to:**
  - `/camera_feed` (raw image)
- **Publishes to:**
  - `/segmentation_mask` (processed segmentation output)
- **Key Features:**
  - Uses **DeepLabV3+** for **semantic segmentation**.
  - Generates color-coded masks for detected objects.

### **🔵 Object Tracking Node** (`object_tracker.py`)
- **Subscribes to:**
  - `/camera_feed` (raw image)
  - `/segmentation_mask` (semantic segmentation mask)
- **Publishes to:**
  - `/tracked_objects` (IDs, class labels, bounding boxes)
- **Key Features:**
  - Extracts objects from the segmentation mask.
  - Assigns a **unique ID** using **DeepSORT Tracker**.
  - Maintains **stable IDs** for objects as long as they remain in the frame.
  - Reuses the **lowest available ID** when an object exits and a new one appears.

### **🟢 Optical Flow Node** (`optical_flow_node.py`)
- **Subscribes to:**
  - `/camera_feed` (raw image)
  - `/tracked_objects` (object bounding boxes)
- **Publishes to:**
  - `/tracked_speeds` (object ID, estimated speed)
- **Key Features:**
  - Uses **Lucas-Kanade Optical Flow** to estimate object speeds.
  - Computes speed in **km/h**.
  - Filters out small movements to reduce noise.

### **🟣 Data Fusion Node** (`data_fusion.py`)
- **Subscribes to:**
  - `/tracked_objects` (object positions)
  - `/tracked_speeds` (object speeds)
- **Publishes to:**
  - `/objects_list` (merged object position & speed data)
- **Key Features:**
  - Merges object tracking and speed data into a unified message.
  - Ensures correct speed association with the **same tracked ID**.

---

## **📜 Message Definitions**
### **📝 Tracked Objects** (`TrackedObjects.msg`)
```yaml
TrackedObject[] objects
```
### **📝 Tracked Object** (`TrackedObject.msg`)
```yaml
int32 id
string class_label
float64 xmin
float64 ymin
float64 xmax
float64 ymax
```
### **📝 Object Speed** (`ObjectSpeed.msg`)
```yaml
int32 id
float32 speed_kmh
```
### **📝 Objects Speeds** (`ObjectsSpeeds.msg`)
```yaml
ObjectSpeed[] objects
```
### **📝 Object Data** (`ObjectData.msg`)
```yaml
int32 id
string class_label
float32 speed_kmh
float64 xmin
float64 ymin
float64 xmax
float64 ymax
```
### **📝 Objects Data** (`ObjectsData.msg`)
```yaml
ObjectData[] objects
```

---

## **👨‍💻 Contributors**
- **MWA, Adham, Mazen, Rewan, and Fayrouz**

---

## **📌 Future Work**
- Improve **speed estimation accuracy** using Kalman Filtering.
- Add **real-world unit calibration** for distance scaling.
- Integrate with a **3D perception pipeline** for depth-based tracking.

---

## **📜 License**
This project is licensed under the **MIT License**. Feel free to modify and improve it!

---

🚀 **Enjoy building with ROS2!**





