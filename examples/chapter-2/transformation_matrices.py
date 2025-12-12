"""
Transformation matrices example
File: examples/chapter-2/transformation_matrices.py

This example demonstrates the use of transformation matrices in robotics,
including rotation matrices, translation vectors, and homogeneous transformation matrices.
"""
import numpy as np
import math


def rotation_x(angle):
    """Create a rotation matrix around the X axis"""
    return np.array([
        [1, 0, 0, 0],
        [0, math.cos(angle), -math.sin(angle), 0],
        [0, math.sin(angle), math.cos(angle), 0],
        [0, 0, 0, 1]
    ])


def rotation_y(angle):
    """Create a rotation matrix around the Y axis"""
    return np.array([
        [math.cos(angle), 0, math.sin(angle), 0],
        [0, 1, 0, 0],
        [-math.sin(angle), 0, math.cos(angle), 0],
        [0, 0, 0, 1]
    ])


def rotation_z(angle):
    """Create a rotation matrix around the Z axis"""
    return np.array([
        [math.cos(angle), -math.sin(angle), 0, 0],
        [math.sin(angle), math.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def translation(x, y, z):
    """Create a translation matrix"""
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])


def homogeneous_transform(translation_vec, rotation_matrix_3x3):
    """Create a homogeneous transformation matrix from translation and rotation"""
    transform = np.eye(4)
    transform[0:3, 0:3] = rotation_matrix_3x3
    transform[0:3, 3] = translation_vec
    return transform


def forward_kinematics(joint_angles):
    """
    Simple forward kinematics for a 2-DOF planar manipulator.
    This function calculates the end-effector position given joint angles.
    """
    # Link lengths
    l1, l2 = 1.0, 0.8  # meters
    
    # Joint angles
    theta1, theta2 = joint_angles
    
    # Calculate end-effector position
    x = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
    y = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)
    
    return np.array([x, y, 0.0])


def main():
    print("Transformation Matrices in Robotics")
    print("=" * 40)
    
    # Example 1: Rotation matrices
    print("\n1. Rotation Matrices:")
    angle = math.pi / 4  # 45 degrees
    
    print(f"Rotation about X by {angle:.2f} radians:")
    rx = rotation_x(angle)
    print(rx)
    
    print(f"\nRotation about Z by {angle:.2f} radians:")
    rz = rotation_z(angle)
    print(rz)
    
    # Example 2: Translation matrix
    print("\n2. Translation Matrix (2, 3, 1):")
    trans = translation(2, 3, 1)
    print(trans)
    
    # Example 3: Homogeneous transformation
    print("\n3. Homogeneous Transformation:")
    # Create a simple rotation matrix (45 degrees around Z)
    rot_matrix = np.array([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1]
    ])
    trans_vector = np.array([1, 2, 3])
    
    homog_transform = homogeneous_transform(trans_vector, rot_matrix)
    print("Homogeneous transformation matrix:")
    print(homog_transform)
    
    # Example 4: Forward kinematics
    print("\n4. Forward Kinematics Example:")
    joint_angles = [math.pi/3, math.pi/6]  # 60 and 30 degrees
    end_effector_pos = forward_kinematics(joint_angles)
    print(f"Joint angles (rad): {joint_angles}")
    print(f"End-effector position: {end_effector_pos}")
    
    # Example 5: Combining transformations
    print("\n5. Combining Transformations:")
    # Transform 1: Rotate 30 degrees around Z, then translate by (1, 0, 0)
    t1 = translation(1, 0, 0) @ rotation_z(math.pi/6)
    
    # Transform 2: Rotate 45 degrees around Y, then translate by (0, 1, 0)
    t2 = translation(0, 1, 0) @ rotation_y(math.pi/4)
    
    # Combined transformation
    combined = t2 @ t1
    print("Combined transformation matrix:")
    print(combined)
    
    print("\nTransformation matrices are fundamental to robotics for:")
    print("- Representing position and orientation of robot links")
    print("- Calculating forward and inverse kinematics")
    print("- Transforming coordinates between different reference frames")
    print("- Planning robot motions in 3D space")


if __name__ == "__main__":
    main()