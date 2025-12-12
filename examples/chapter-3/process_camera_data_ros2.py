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