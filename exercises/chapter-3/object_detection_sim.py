"""
Object detection in simulated environment exercise
File: exercises/chapter-3/object_detection_sim.py

This exercise demonstrates using PyBullet for physics simulation and
OpenCV for object detection. It creates a simple scene with objects
and detects them using basic computer vision techniques.
"""

import pybullet as p
import pybullet_data
import cv2
import numpy as np
import time
import math


def create_simulated_scene():
    """
    Create a simulated scene with various objects using PyBullet
    """
    # Connect to PyBullet physics server
    physicsClient = p.connect(p.DIRECT)  # Use p.GUI for visualization
    
    # Set gravity
    p.setGravity(0, 0, -10)
    
    # Load plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")
    
    # Create some objects in the scene
    # Red box
    boxStartPos = [1, 0, 1]
    boxStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    boxId = p.loadURDF("cube.urdf", boxStartPos, boxStartOrientation, 
                       globalScaling=0.5)
    p.changeVisualShape(boxId, -1, rgbaColor=[1, 0, 0, 1])  # Red
    
    # Green sphere
    sphereStartPos = [-1, 0.5, 1.2]
    sphereId = p.loadURDF("sphere.urdf", sphereStartPos, 
                          globalScaling=0.3)
    p.changeVisualShape(sphereId, -1, rgbaColor=[0, 1, 0, 1])  # Green
    
    # Blue cylinder
    cylinderStartPos = [0, -1, 1.5]
    cylinderStartOrientation = p.getQuaternionFromEuler([math.pi/2, 0, 0])
    cylinderId = p.loadURDF("cylinder.urdf", cylinderStartPos, 
                            cylinderStartOrientation, 
                            globalScaling=0.4)
    p.changeVisualShape(cylinderId, -1, rgbaColor=[0, 0, 1, 1])  # Blue
    
    # Set camera parameters
    cameraDistance = 3
    cameraYaw = 45
    cameraPitch = -30
    cameraTargetPosition = [0, 0, 1]
    
    return physicsClient, boxId, sphereId, cylinderId, cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition


def get_camera_image(physicsClient, cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition):
    """
    Get an image from the simulated camera
    """
    # Get camera image
    viewMatrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition, cameraDistance, cameraYaw, cameraPitch, 0, 2)
    
    # Set projection matrix
    aspect = 1.0
    projectionMatrix = p.computeProjectionMatrixFOV(45, aspect, 0.1, 3.1)
    
    # Get image from the camera
    width, height, rgbaImg, depthImg, segImg = p.getCameraImage(
        width=640, height=480, viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix, 
        renderer=p.ER_BULLET_HARDWARE_OPENGL)
    
    # Convert RGBA image to BGR for OpenCV
    rgb_array = np.array(rgbaImg)
    rgb_array = np.reshape(rgb_array, (height, width, 4))
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
    
    return bgr_array


def detect_objects(image):
    """
    Detect objects in the image using basic computer vision techniques
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for red color
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2
    
    # Define range for green color
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Define range for blue color
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Find contours for each color
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    result_image = image.copy()
    
    # Draw red contours
    cv2.drawContours(result_image, contours_red, -1, (0, 0, 255), 2)
    for contour in contours_red:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(result_image, "Red Object", (cx, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw green contours
    cv2.drawContours(result_image, contours_green, -1, (0, 255, 0), 2)
    for contour in contours_green:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(result_image, "Green Object", (cx, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw blue contours
    cv2.drawContours(result_image, contours_blue, -1, (255, 0, 0), 2)
    for contour in contours_blue:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(result_image, "Blue Object", (cx, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return result_image, contours_red, contours_green, contours_blue


def main():
    """
    Main function to run the object detection exercise
    """
    print("Starting object detection in simulated environment...")
    
    # Create the simulated scene
    physicsClient, boxId, sphereId, cylinderId, cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition = create_simulated_scene()
    
    # Get image from simulated camera
    image = get_camera_image(physicsClient, cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)
    
    # Detect objects in the image
    result_image, red_contours, green_contours, blue_contours = detect_objects(image)
    
    # Count detected objects
    num_red = len(red_contours)
    num_green = len(green_contours)
    num_blue = len(blue_contours)
    
    print(f"Detected {num_red} red objects, {num_green} green objects, {num_blue} blue objects")
    
    # Show the original and processed images
    cv2.imshow("Original Image", image)
    cv2.imshow("Detected Objects", result_image)
    
    # Wait for key press to close windows
    print("Press any key on the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Disconnect from physics server
    p.disconnect()
    
    print("Object detection exercise completed!")


if __name__ == "__main__":
    main()