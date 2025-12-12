---
sidebar_position: 2
---

# Mathematical Foundations for Robotics

## Learning Objectives

After reading this chapter, you will be able to:
- Apply linear algebra concepts to robotic transformations
- Understand kinematic equations and coordinate systems
- Calculate Jacobian matrices and understand their role in velocity kinematics
- Implement transformation matrices in Python
- Solve forward kinematics problems for simple robotic arms

## Prerequisites

Before reading this chapter, you should:
- Have a basic understanding of linear algebra (vectors, matrices, matrix multiplication)
- Be familiar with calculus concepts (derivatives)
- Have basic Python programming skills

## Introduction

Robotics relies heavily on mathematics to model, control, and understand robotic systems. From describing the position and orientation of a robot's end-effector to calculating the forces and velocities throughout the system, mathematics provides the language for roboticists to design and operate robots effectively.

This chapter covers the fundamental mathematical concepts used in robotics, with a focus on topics most relevant to robotics applications. We'll start with transformation matrices, which allow us to describe the position and orientation of objects in 3D space, then move to kinematic equations that relate joint angles to the position of the robot's end-effector.

## Linear Algebra for Robotics

### Vectors and Vector Operations

In robotics, we use vectors to represent positions, velocities, forces, and other quantities that have both magnitude and direction.

```python
import numpy as np

# Position vector in 3D space
position = np.array([1.0, 2.0, 3.0])
print(f"Position vector: {position}")

# Vector operations
vector_a = np.array([1.0, 0.0, 1.0])
vector_b = np.array([0.0, 1.0, 1.0])

# Addition
vector_sum = vector_a + vector_b
print(f"Vector sum: {vector_sum}")

# Magnitude (norm)
magnitude_a = np.linalg.norm(vector_a)
print(f"Magnitude of vector_a: {magnitude_a}")

# Dot product (for angle calculations)
dot_product = np.dot(vector_a, vector_b)
print(f"Dot product: {dot_product}")

# Cross product (for perpendicular vectors)
cross_product = np.cross(vector_a, vector_b)
print(f"Cross product: {cross_product}")
```

### Transformation Matrices

One of the most important concepts in robotics is the transformation matrix, which combines rotation and translation into a single 4x4 matrix. This matrix transforms points from one coordinate system to another.

#### Homogeneous Transformation

```python
def create_translation_matrix(tx, ty, tz):
    """
    Create a 4x4 translation matrix.
    
    Args:
        tx, ty, tz: Translation along x, y, z axes respectively
    
    Returns:
        4x4 transformation matrix
    """
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

def create_rotation_x_matrix(angle_rad):
    """
    Create a 4x4 rotation matrix around the X-axis.
    
    Args:
        angle_rad: Rotation angle in radians
    
    Returns:
        4x4 rotation matrix
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])

def create_rotation_z_matrix(angle_rad):
    """
    Create a 4x4 rotation matrix around the Z-axis.
    
    Args:
        angle_rad: Rotation angle in radians
    
    Returns:
        4x4 rotation matrix
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def transform_point(point, transform_matrix):
    """
    Transform a 3D point using a 4x4 transformation matrix.
    
    Args:
        point: 3-element vector representing the point
        transform_matrix: 4x4 transformation matrix
    
    Returns:
        Transformed 3D point
    """
    # Convert to homogeneous coordinates
    point_h = np.append(point, 1)
    
    # Apply transformation
    transformed_h = transform_matrix @ point_h
    
    # Convert back to 3D coordinates
    return transformed_h[:3]

# Example usage
print("Transformation Matrix Example:")
print("=" * 35)

# Create a transformation matrix (translate by [1, 2, 3] and rotate 45° around Z)
translation = create_translation_matrix(1, 2, 3)
rotation_z = create_rotation_z_matrix(np.pi / 4)  # 45 degrees in radians

# Combine transformations (order matters: rotation first, then translation)
combined_transform = translation @ rotation_z

# Transform a point
original_point = np.array([1.0, 0.0, 0.0])  # Point at (1, 0, 0)
transformed_point = transform_point(original_point, combined_transform)

print(f"Original point: {original_point}")
print(f"Transformed point: {transformed_point}")
```

## Coordinate Systems and Representations

### Euler Angles

Euler angles represent orientation using three angles of rotation around the X, Y, and Z axes. The order of rotations matters and there are multiple conventions (e.g., ZYX, XYZ).

```python
def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a rotation matrix.
    Uses ZYX convention (intrinsic rotations).
    
    Args:
        roll: Rotation angle around X-axis (radians)
        pitch: Rotation angle around Y-axis (radians)  
        yaw: Rotation angle around Z-axis (radians)
    
    Returns:
        3x3 rotation matrix
    """
    # Rotation around Z (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Rotation around Y (pitch)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Rotation around X (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Combined rotation matrix (R = Rz * Ry * Rx)
    return Rz @ Ry @ Rx

# Example: Rotate by 30° around X, 45° around Y, 60° around Z
roll = np.radians(30)
pitch = np.radians(45)
yaw = np.radians(60)

rotation_matrix = euler_to_rotation_matrix(roll, pitch, yaw)
print(f"Rotation matrix from Euler angles:\n{rotation_matrix}")
```

### Quaternions

Quaternions are another way to represent rotations that avoid some issues with Euler angles like gimbal lock.

```python
class Quaternion:
    """
    A simple quaternion class for representing rotations.
    """
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w = w  # Scalar part
        self.x = x  # i component
        self.y = y  # j component
        self.z = z  # k component
    
    def normalize(self):
        """Normalize the quaternion."""
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm > 0:
            self.w /= norm
            self.x /= norm
            self.y /= norm
            self.z /= norm
    
    def to_rotation_matrix(self):
        """Convert quaternion to rotation matrix."""
        w, x, y, z = self.w, self.x, self.y, self.z
        
        # Ensure quaternion is normalized
        self.normalize()
        
        # Calculate matrix elements
        m00 = 1 - 2*(y**2 + z**2)
        m01 = 2*(x*y - z*w)
        m02 = 2*(x*z + y*w)
        
        m10 = 2*(x*y + z*w)
        m11 = 1 - 2*(x**2 + z**2)
        m12 = 2*(y*z - x*w)
        
        m20 = 2*(x*z - y*w)
        m21 = 2*(y*z + x*w)
        m22 = 1 - 2*(x**2 + y**2)
        
        return np.array([
            [m00, m01, m02],
            [m10, m11, m12],
            [m20, m21, m22]
        ])
    
    @classmethod
    def from_axis_angle(cls, axis, angle):
        """
        Create quaternion from axis-angle representation.
        
        Args:
            axis: 3-element unit vector representing rotation axis
            angle: Rotation angle in radians
        """
        axis = np.array(axis) / np.linalg.norm(axis)  # Normalize axis
        half_angle = angle / 2
        w = np.cos(half_angle)
        x, y, z = np.sin(half_angle) * axis
        return cls(w, x, y, z)

# Example: Rotation of 90 degrees around the Z-axis
quat = Quaternion.from_axis_angle([0, 0, 1], np.pi/2)  # 90 degrees
rotation_matrix = quat.to_rotation_matrix()
print(f"Rotation matrix from quaternion:\n{rotation_matrix}")
```

## Kinematic Equations

### Forward Kinematics

Forward kinematics calculates the position and orientation of the end-effector given the joint angles. For a simple 2D robotic arm:

```python
def forward_kinematics_2d(joint_angles, link_lengths):
    """
    Calculate the end-effector position of a 2D robotic arm.
    
    Args:
        joint_angles: List of joint angles in radians
        link_lengths: List of link lengths
    
    Returns:
        (x, y) position of end-effector
    """
    x = 0.0
    y = 0.0
    cumulative_angle = 0.0
    
    for angle, length in zip(joint_angles, link_lengths):
        cumulative_angle += angle
        x += length * np.cos(cumulative_angle)
        y += length * np.sin(cumulative_angle)
    
    return x, y

def forward_kinematics_3d_2link(theta1, theta2, l1, l2):
    """
    Calculate the 3D position of a 2-link robot arm (assuming in x-y plane).
    
    Args:
        theta1: Angle of first joint (base rotation)
        theta2: Angle of second joint (elbow rotation)
        l1: Length of first link
        l2: Length of second link
    
    Returns:
        (x, y, z) position of end-effector
    """
    # Calculate position relative to base
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    z = 0  # On x-y plane
    
    return x, y, z

# Example: 2-link planar arm
l1 = 1.0  # Link 1 length
l2 = 0.8  # Link 2 length
theta1 = np.pi/3  # 60 degrees
theta2 = np.pi/4  # 45 degrees

end_effector_pos = forward_kinematics_3d_2link(theta1, theta2, l1, l2)
print(f"End-effector position: ({end_effector_pos[0]:.2f}, {end_effector_pos[1]:.2f}, {end_effector_pos[2]:.2f})")
```

### The Jacobian Matrix

The Jacobian matrix relates joint velocities to end-effector velocities:

```python
def jacobian_2d_planar(joint_angles, link_lengths):
    """
    Calculate the geometric Jacobian for a 2D planar manipulator.
    
    Args:
        joint_angles: List of joint angles [theta1, theta2, ...]
        link_lengths: List of link lengths [l1, l2, ...]
    
    Returns:
        2xN Jacobian matrix (N = number of joints)
    """
    n = len(joint_angles)
    jacobian = np.zeros((2, n))  # [dx; dy] / [dtheta1; dtheta2; ...]
    
    # Calculate end-effector position from each joint to the end
    total_x = 0.0
    total_y = 0.0
    
    # For each column of the Jacobian (each joint)
    for i in range(n):
        # Calculate position of end-effector relative to joint i
        x_to_end = 0.0
        y_to_end = 0.0
        for j in range(i, n):
            # Sum all links from joint i to end
            angle_to_j = sum(joint_angles[i:j+1])
            x_to_end += link_lengths[j] * np.cos(angle_to_j)
            y_to_end += link_lengths[j] * np.sin(angle_to_j)
        
        # Calculate Jacobian elements: J_row_i = [∂x/∂theta_i, ∂y/∂theta_i]
        jacobian[0, i] = -y_to_end  # dx/dtheta_i
        jacobian[1, i] = x_to_end   # dy/dtheta_i
    
    return jacobian

def jacobian_2link_planar(theta1, theta2, l1, l2):
    """
    Calculate the Jacobian for a 2-link planar manipulator.
    
    Args:
        theta1: Angle of first joint
        theta2: Angle of second joint
        l1: Length of first link
        l2: Length of second link
    
    Returns:
        2x2 Jacobian matrix
    """
    # J = [Jv, Jw] where Jv is linear velocity part, Jw is angular velocity part
    # For a planar manipulator, angular velocity is in the z direction
    jacobian = np.zeros((2, 2))
    
    # Linear velocity part
    jacobian[0, 0] = -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)  # ∂x/∂theta1
    jacobian[0, 1] = -l2 * np.sin(theta1 + theta2)                        # ∂x/∂theta2
    jacobian[1, 0] = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)    # ∂y/∂theta1
    jacobian[1, 1] = l2 * np.cos(theta1 + theta2)                         # ∂y/∂theta2
    
    return jacobian

# Example: Calculate Jacobian for 2-link arm
jacobian = jacobian_2link_planar(theta1, theta2, l1, l2)
print(f"Jacobian matrix:\n{jacobian}")

# Example: Use Jacobian to find end-effector velocity given joint velocities
joint_velocities = np.array([0.1, 0.2])  # [theta1_dot, theta2_dot]
end_effector_velocity = jacobian @ joint_velocities
print(f"End-effector velocity: ({end_effector_velocity[0]:.3f}, {end_effector_velocity[1]:.3f})")
```

## Implementation Example: Transformations

```python
import math

class Transform3D:
    """
    A class to represent 3D transformations combining rotation and translation.
    """
    def __init__(self, translation=None, rotation_matrix=None):
        """
        Initialize a 3D transformation.
        
        Args:
            translation: 3-element array for translation [tx, ty, tz]
            rotation_matrix: 3x3 rotation matrix
        """
        self.translation = np.zeros(3) if translation is None else np.array(translation)
        self.rotation = np.eye(3) if rotation_matrix is None else rotation_matrix
    
    def to_homogeneous_matrix(self):
        """Convert to 4x4 homogeneous transformation matrix."""
        matrix = np.eye(4)
        matrix[0:3, 0:3] = self.rotation
        matrix[0:3, 3] = self.translation
        return matrix
    
    def transform_point(self, point):
        """Transform a 3D point."""
        # Apply rotation then translation
        rotated = self.rotation @ point
        return rotated + self.translation
    
    def compose(self, other_transform):
        """
        Compose this transformation with another.
        This @ Other (apply other first, then this).
        """
        new_rotation = self.rotation @ other_transform.rotation
        new_translation = self.rotation @ other_transform.translation + self.translation
        return Transform3D(new_translation, new_rotation)
    
    def inverse(self):
        """Calculate the inverse transformation."""
        inv_rotation = self.rotation.T
        inv_translation = -inv_rotation @ self.translation
        return Transform3D(inv_translation, inv_rotation)

# Example: Chain of transformations
print("Example: Chain of transformations")
print("-" * 35)

# Define transformations
t1 = Transform3D([1, 0, 0])  # Translate by 1 unit in x
t2 = Transform3D(
    translation=[0, 1, 0],
    rotation_matrix=euler_to_rotation_matrix(0, 0, np.pi/4)  # 45° around z
)

# Compose transformations
combined = t1.compose(t2)

# Transform a point
original_point = np.array([1, 0, 0])
transformed_point = combined.transform_point(original_point)

print(f"Original point: {original_point}")
print(f"After transformation: {transformed_point}")

# Verify with homogeneous matrix
homogeneous_matrix = combined.to_homogeneous_matrix()
point_h = np.append(original_point, 1)
transformed_h = homogeneous_matrix @ point_h
transformed_with_matrix = transformed_h[:3]

print(f"With homogeneous matrix: {transformed_with_matrix}")
print(f"Match: {np.allclose(transformed_point, transformed_with_matrix)}")
```

## Summary

This chapter covered the essential mathematical foundations for robotics:

1. **Linear algebra concepts** - vectors, matrices, transformations
2. **Transformation matrices** - combining rotation and translation
3. **Coordinate system representations** - Euler angles, quaternions
4. **Kinematic equations** - forward kinematics and the Jacobian matrix
5. **Practical implementations** - using Python and NumPy for calculations

These mathematical tools are fundamental to understanding and controlling robotic systems. They allow us to model the robot's configuration, calculate how it moves, and plan movements to achieve desired positions and orientations.

## Exercises

1. Implement a function to convert a rotation matrix to Euler angles (implement the inverse of the `euler_to_rotation_matrix` function)
2. Create a class for a 6-DOF robotic arm and implement its forward kinematics
3. Derive the Jacobian matrix for a 3-link planar manipulator
4. Implement quaternion multiplication and interpolation (SLERP)

## References

- Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2006). *Robot Modeling and Control*. Wiley.
- Craig, J. J. (2005). *Introduction to Robotics: Mechanics and Control* (3rd ed.). Pearson.
- Siciliano, B., & Khatib, O. (Eds.). (2016). *Springer Handbook of Robotics* (2nd ed.). Springer.