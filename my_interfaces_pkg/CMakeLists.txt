cmake_minimum_required(VERSION 3.5)
project(my_interfaces_pkg)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(action_msgs REQUIRED)

# Add these lines for action generation
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
"msg/StartMissionMsg.msg"
"msg/Ticks.msg"
"msg/TrackedObject.msg"
"msg/TrackedObjects.msg"
"msg/ObjectSpeed.msg"
"msg/ObjectsSpeeds.msg"
"msg/ObjectData.msg"
"msg/ObjectsData.msg"
"srv/StartMission.srv"
"srv/CreatePlan.srv" 
"action/CountUntil.action"
"action/Navigate.action"
  DEPENDENCIES action_msgs geometry_msgs nav_msgs
)

ament_export_dependencies(rosidl_default_runtime)


ament_package()
