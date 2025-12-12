---
title: Code Examples Tutorial
sidebar_position: 3
---

# Code Examples Tutorial

This tutorial explains how to work with the code examples in this book on Physical AI and Humanoid Robotics. It covers the structure of examples, how to run them, and best practices for experimentation.

## Overview of Example Structure

The code examples in this book are organized by chapter in the following directory structure:

```
examples/
├── chapter-1/
│   ├── basic_robot_model.py
│   └── ...
├── chapter-2/
│   ├── transformation_matrices.py
│   └── ...
├── chapter-3/
│   ├── process_camera_data_ros2.py
│   └── ...
├── ...
└── chapter-12/
    └── safety_controller.py
```

Each chapter directory contains examples that demonstrate the concepts discussed in that chapter.

## Running Examples

### Prerequisites

Before running the examples, ensure you have:

1. Set up your Python environment as described in the Setup Environment Tutorial
2. Installed all required packages from `requirements.txt`
3. Activated your virtual environment

```bash
source physical_ai_env/bin/activate  # On Windows: physical_ai_env\Scripts\activate
```

### How to Execute Examples

1. **Navigate to the example directory:**
   ```bash
   cd examples/chapter-8  # For Chapter 8 examples
   ```

2. **Run a specific example:**
   ```bash
   python learning_from_demo.py
   ```

3. **For Jupyter notebooks (in the `notebooks/` directory):**
   ```bash
   jupyter notebook
   # Then open the notebook in your browser
   ```

## Example: Processing Camera Data in ROS 2

Let's walk through one of the examples from Chapter 3: `process_camera_data_ros2.py`

```python
"""
Processing camera data in ROS 2
File: examples/chapter-3/process_camera_data_ros2.py

This example demonstrates how to subscribe to camera data in ROS 2,
process the image data, and publish results. This simulates the kind
of camera processing needed in humanoid robots for perception tasks.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np


class CameraDataProcessor(Node):
    """
    A ROS 2 node that subscribes to camera data, processes images,
    and publishes processed results.
    """
    
    def __init__(self):
        super().__init__('camera_data_processor')
        
        # Create subscription to camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.subscription  # Prevent unused variable warning
        
        # Create publisher for processed results
        self.result_publisher = self.create_publisher(String, '/camera_processing_result', 10)
        
        # Initialize OpenCV bridge
        self.bridge = CvBridge()
        
        # Processed frame counter
        self.frame_count = 0
        
        self.get_logger().info('Camera Data Processor node initialized')

    def image_callback(self, msg):
        """
        Callback function to process incoming camera images
        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Process the image
            processed_image = self.process_image(cv_image)
            
            # Analyze the image
            analysis_result = self.analyze_image(cv_image)
            
            # Publish the analysis result
            result_msg = String()
            result_msg.data = f"Frame {self.frame_count}: {analysis_result}"
            self.result_publisher.publish(result_msg)
            
            self.frame_count += 1
            self.get_logger().info(f'Processed frame {self.frame_count}: {analysis_result}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def process_image(self, image):
        """
        Perform basic image processing operations
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges using Canny edge detector
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on the original image
        result_image = image.copy()
        cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
        
        return result_image

    def analyze_image(self, image):
        """
        Perform basic image analysis and return results
        """
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for red color (common in humanoid robot scenarios)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        
        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        
        # Combine masks
        mask = mask1 + mask2
        
        # Count red pixels
        red_pixels = cv2.countNonZero(mask)
        total_pixels = image.shape[0] * image.shape[1]
        red_percentage = (red_pixels / total_pixels) * 100
        
        # Detect edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = cv2.countNonZero(edges)
        edge_percentage = (edge_pixels / total_pixels) * 100
        
        # Return analysis result
        return f"Red pixels: {red_percentage:.2f}%, Edges: {edge_percentage:.2f}%"

    def run_demo(self):
        """
        Run a demonstration of camera processing
        """
        # In a real scenario, we would simulate camera data or connect to a real camera
        # For this example, we'll just log what would happen
        self.get_logger().info('Starting camera processing demo...')
        self.get_logger().info('Subscribed to /camera/image_raw')
        self.get_logger().info('Publishing results to /camera_processing_result')


def main(args=None):
    """
    Main function to run the camera data processor node
    """
    rclpy.init(args=args)
    
    camera_processor = CameraDataProcessor()
    
    try:
        camera_processor.run_demo()
        rclpy.spin(camera_processor)
    except KeyboardInterrupt:
        camera_processor.get_logger().info('Interrupted, shutting down...')
    finally:
        camera_processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Running the Example

To run this example in a simulated environment:

```bash
cd examples/chapter-3
python process_camera_data_ros2.py
```

## Example: Safety Controller

Let's examine another example from Chapter 12: `safety_controller.py`

```python
"""
Safety controller implementation
File: examples/chapter-12/safety_controller.py

This example implements a safety controller for humanoid robots that monitors
critical parameters and ensures the robot remains in safe operational states.
The controller implements multiple safety layers including joint limits,
velocity limits, and collision avoidance.
"""

import numpy as np
import time
import threading
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


class SafetyState(Enum):
    NORMAL = 1
    WARNING = 2
    EMERGENCY_STOP = 3
    SAFE_STATE = 4


@dataclass
class RobotState:
    """Represents the current state of the robot"""
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    joint_names: List[str]
    robot_position: np.ndarray  # x, y, z position in world
    robot_orientation: np.ndarray  # quaternion [x, y, z, w]
    imu_data: np.ndarray  # [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    proximity_sensors: np.ndarray  # distances to obstacles
    external_forces: np.ndarray  # forces applied to robot
    joint_temperatures: np.ndarray  # temperatures for each joint


class SafetyController:
    """
    A comprehensive safety controller for humanoid robots implementing
    multiple safety layers and protocols.
    """
    
    def __init__(self):
        # Safety configuration
        self.safety_state = SafetyState.NORMAL
        self.emergency_stop_triggered = False
        
        # Joint safety limits
        self.joint_position_limits = {
            'hip_pitch': (-1.5, 1.5),
            'knee_pitch': (0.0, 2.5),
            'ankle_pitch': (-0.5, 0.5),
            'hip_roll': (-0.5, 0.5),
            'ankle_roll': (-0.5, 0.5),
            'shoulder_pitch': (-2.0, 2.0),
            'shoulder_roll': (-1.5, 1.0),
            'elbow_pitch': (-2.0, 0.5),
        }
        
        self.velocity_limit = 2.0  # rad/s
        self.torque_limits = {
            'hip': 100.0,    # Nm
            'knee': 80.0,
            'ankle': 40.0,
            'shoulder': 60.0,
            'elbow': 30.0,
        }
        
        self.proximity_threshold = 0.5  # meters
        self.temperature_threshold = 75.0  # Celsius
        self.fall_threshold = 0.3  # Radians from upright
        
        # Safe position to move to in emergency
        self.safe_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Simplified
        
        # Logging
        self.safety_log = []
        
    def check_joint_limits(self, state: RobotState) -> Tuple[bool, str]:
        """Check if all joint positions are within safe limits"""
        for i, name in enumerate(state.joint_names):
            if name in self.joint_position_limits:
                pos = state.joint_positions[i]
                min_limit, max_limit = self.joint_position_limits[name]
                
                if pos < min_limit or pos > max_limit:
                    msg = f"Joint {name} position limit violated: {pos:.3f} (limits: {min_limit:.3f}, {max_limit:.3f})"
                    return False, msg
        
        return True, "All joint positions within limits"
    
    def check_velocity_limits(self, state: RobotState) -> Tuple[bool, str]:
        """Check if all joint velocities are within safe limits"""
        max_velocity = np.max(np.abs(state.joint_velocities))
        
        if max_velocity > self.velocity_limit:
            return False, f"Velocity limit exceeded: {max_velocity:.3f} rad/s (limit: {self.velocity_limit})"
        
        return True, "All velocities within limits"
    
    # Additional methods would be here in the full example
    # ... (other safety checks)
    
    def run_safety_checks(self, robot_state: RobotState) -> Tuple[SafetyState, List[str]]:
        """
        Run all safety checks on the provided robot state
        Returns the safety state and a list of safety messages
        """
        # Perform all safety checks
        checks = [
            self.check_joint_limits(robot_state),
            self.check_velocity_limits(robot_state),
            # ... other checks would be here
        ]
        
        # Evaluate results
        all_safe = True
        safety_messages = []
        
        for is_safe, message in checks:
            if not is_safe:
                all_safe = False
                safety_messages.append(message)
        
        # Update safety state based on results
        if not all_safe:
            if self.safety_state == SafetyState.NORMAL:
                self.safety_state = SafetyState.WARNING
            elif self.safety_state == SafetyState.WARNING:
                self.safety_state = SafetyState.EMERGENCY_STOP
            elif self.safety_state in [SafetyState.EMERGENCY_STOP, SafetyState.SAFE_STATE]:
                self.safety_state = SafetyState.SAFE_STATE
        else:
            # If all checks pass, reset to normal state (if not in safe state from previous emergency)
            if self.safety_state != SafetyState.SAFE_STATE:
                self.safety_state = SafetyState.NORMAL
            safety_messages = ["All safety checks passed"]
        
        # Log safety state changes
        self.safety_log.append({
            'timestamp': time.time(),
            'state': self.safety_state,
            'messages': safety_messages.copy()
        })
        
        return self.safety_state, safety_messages

# The rest of the implementation would continue here...
```

## Running Example with Jupyter Notebooks

For notebook examples, you'll find them in the `notebooks/` directory:

```bash
cd notebooks/chapter-12
jupyter notebook safety_controller.ipynb
```

This will open the notebook in your browser where you can run the cells interactively.

## Modifying Examples

The examples are designed to be modified and experimented with. Here are some suggestions:

1. **Change Parameters**: Modify values like velocities, thresholds, or control gains
2. **Add Visualization**: Add plots or graphical displays of data
3. **Try Different Algorithms**: Replace one algorithm with another to see differences
4. **Combine Examples**: Combine elements from multiple examples to create new functionality

For example, you could modify the safety controller example to add temperature monitoring:

```python
# In the SafetyController class
def check_temperature_safety(self, state: RobotState) -> Tuple[bool, str]:
    """Check if all joint temperatures are within safe limits"""
    max_temp = np.max(state.joint_temperatures)
    
    if max_temp > self.temperature_threshold:
        return False, f"Temperature limit exceeded: {max_temp:.1f}°C (threshold: {self.temperature_threshold}°C)"
    
    return True, "All temperatures within limits"
```

## Troubleshooting Examples

If an example doesn't run correctly:

1. **Check Dependencies**: Ensure all required packages are installed
2. **Review Error Messages**: Read the error message carefully
3. **Check Python Version**: Some examples require specific Python versions
4. **Virtual Environment**: Make sure you're in the correct virtual environment

### Common Issues:

- **Missing package**: Run `pip install <package_name>`
- **ROS not found**: Run `source /opt/ros/humble/setup.bash` (or appropriate ROS version)
- **PyBullet error**: Reinstall with `pip install --upgrade --force-reinstall pybullet`

## Extending Examples

Each example is a starting point for more complex implementations:

1. **Add Error Handling**: Implement more robust error handling
2. **Improve Performance**: Optimize algorithms for better performance
3. **Add Features**: Extend functionality based on your needs
4. **Create Variants**: Develop multiple versions with different approaches

## Best Practices

1. **Version Control**: Use git to track changes to examples
2. **Documentation**: Add comments explaining what your modifications do
3. **Testing**: Test your modifications thoroughly
4. **Backup**: Keep originals of examples before extensive modifications

## Summary

The code examples in this book are designed to be educational tools that you can:

- Run as-is to see concepts in action
- Modify to understand how different parameters affect behavior
- Extend to build more complex functionality
- Use as starting points for your own projects

These examples form the practical foundation for understanding Physical AI and Humanoid Robotics concepts discussed in the book.