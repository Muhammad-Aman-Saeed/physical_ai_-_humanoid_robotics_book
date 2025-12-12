---
sidebar_position: 4
---

# Forward and Inverse Kinematics

## Learning Objectives

After reading this chapter, you will be able to:
- Calculate forward kinematics for robotic manipulators using the Denavit-Hartenberg (DH) convention
- Understand and implement inverse kinematics solutions for common robot configurations
- Analyze workspace and dexterity of robotic manipulators
- Implement both analytical and numerical solutions for inverse kinematics
- Apply kinematic constraints and joint limits in robotic systems

## Prerequisites

Before reading this chapter, you should:
- Understand basic linear algebra and transformation matrices (covered in Chapter 2)
- Be familiar with coordinate systems and rotation matrices
- Have basic Python programming skills

## Introduction

Kinematics is the study of motion without considering the forces that cause the motion. In robotics, kinematics describes the relationship between the joint angles of a robot and the position and orientation of its end-effector. This relationship is fundamental to robot control, as it allows us to determine how to move the joints to achieve a desired end-effector pose, or conversely, to determine the end-effector pose based on the current joint configuration.

Forward kinematics and inverse kinematics are the two fundamental problems in robot kinematics:

1. **Forward kinematics**: Given the joint variables, find the end-effector position and orientation
2. **Inverse kinematics**: Given the desired end-effector position and orientation, find the required joint variables

## Forward Kinematics

Forward kinematics involves calculating the position and orientation of the end-effector given the joint angles. This is typically done using transformation matrices that describe the relationship between consecutive links in the robot.

### The Denavit-Hartenberg (DH) Convention

The DH convention is a systematic approach to assigning coordinate frames to the joints and links of a robot. This convention defines four parameters for each joint:

- **a_i**: Link length (distance along x_i from z_(i-1) to z_i)
- **α_i**: Link twist (angle between z_(i-1) and z_i measured about x_i)
- **d_i**: Link offset (distance along z_(i-1) from x_(i-1) to x_i)
- **θ_i**: Joint angle (angle between x_(i-1) and x_i measured about z_(i-1))

```python
import numpy as np

class DHParameters:
    """
    Class to represent Denavit-Hartenberg parameters for a single joint.
    """
    def __init__(self, a, alpha, d, theta):
        self.a = a        # Link length
        self.alpha = alpha  # Link twist
        self.d = d        # Link offset
        self.theta = theta  # Joint angle
    
    def to_transformation_matrix(self):
        """
        Convert DH parameters to homogeneous transformation matrix.
        
        Returns:
            4x4 transformation matrix
        """
        # Calculate the transformation matrix using DH parameters
        # T = Rot(z, θ_i) * Trans(z, d_i) * Trans(x, a_i) * Rot(x, α_i)
        
        cos_th = np.cos(self.theta)
        sin_th = np.sin(self.theta)
        cos_al = np.cos(self.alpha)
        sin_al = np.sin(self.alpha)
        
        return np.array([
            [cos_th, -sin_th*cos_al, sin_th*sin_al, self.a*cos_th],
            [sin_th, cos_th*cos_al, -cos_th*sin_al, self.a*sin_th],
            [0, sin_al, cos_al, self.d],
            [0, 0, 0, 1]
        ])

def calculate_forward_kinematics(dh_params_list, joint_angles):
    """
    Calculate forward kinematics for a serial robot using DH parameters.
    
    Args:
        dh_params_list: List of DHParameters objects for each joint
        joint_angles: List of joint angles (for revolute joints)
    
    Returns:
        4x4 final transformation matrix
    """
    if len(dh_params_list) != len(joint_angles):
        raise ValueError("Number of DH parameters must match number of joint angles")
    
    # Initialize total transformation as identity matrix
    T_total = np.eye(4)
    
    # Multiply transformation matrices for each joint
    for i, dh_params in enumerate(dh_params_list):
        # Update joint angle for revolute joints
        current_dh = DHParameters(
            dh_params.a, 
            dh_params.alpha, 
            dh_params.d, 
            joint_angles[i]
        )
        T_joint = current_dh.to_transformation_matrix()
        T_total = T_total @ T_joint
    
    return T_total

# Example: PUMA 560 robot (simplified DH parameters)
# Based on standard DH parameters for PUMA 560
puma_dh_params = [
    DHParameters(0, -np.pi/2, 0, 0),      # Joint 1
    DHParameters(0.4318, 0, 0, 0),        # Joint 2
    DHParameters(-0.0203, -np.pi/2, 0.15005, 0),  # Joint 3
    DHParameters(0, np.pi/2, 0.4318, 0),  # Joint 4
    DHParameters(0, -np.pi/2, 0, 0),      # Joint 5
    DHParameters(0, 0, 0, 0)              # Joint 6
]

# Calculate forward kinematics for some joint angles
joint_angles = [0, np.pi/4, -np.pi/4, 0, np.pi/2, 0]
end_effector_pose = calculate_forward_kinematics(puma_dh_params, joint_angles)

print("End-effector pose (4x4 transformation matrix):")
print(end_effector_pose)
print(f"End-effector position: ({end_effector_pose[0,3]:.3f}, {end_effector_pose[1,3]:.3f}, {end_effector_pose[2,3]:.3f})")
```

### Forward Kinematics for Planar Manipulators

For simpler planar manipulators, we can compute forward kinematics directly:

```python
def forward_kinematics_planar(joint_angles, link_lengths):
    """
    Calculate forward kinematics for a planar manipulator.
    
    Args:
        joint_angles: List of joint angles [theta1, theta2, ...]
        link_lengths: List of link lengths [l1, l2, ...]
    
    Returns:
        List of (x, y) coordinates for each joint position
    """
    if len(joint_angles) != len(link_lengths):
        raise ValueError("Number of joint angles must match number of link lengths")
    
    joint_positions = [(0, 0)]  # Base position
    cumulative_angle = 0.0
    
    for i, (angle, length) in enumerate(zip(joint_angles, link_lengths)):
        cumulative_angle += angle
        
        # Calculate position of current joint
        prev_x, prev_y = joint_positions[-1]
        x = prev_x + length * np.cos(cumulative_angle)
        y = prev_y + length * np.sin(cumulative_angle)
        
        joint_positions.append((x, y))
    
    return joint_positions

# Example: 3-link planar manipulator
link_lengths = [1.0, 0.8, 0.6]
joint_angles = [np.pi/4, -np.pi/6, np.pi/3]

joint_positions = forward_kinematics_planar(joint_angles, link_lengths)

print("\nPlanar Manipulator Forward Kinematics:")
print("-" * 40)
for i, (x, y) in enumerate(joint_positions):
    print(f"Joint {i}: ({x:.3f}, {y:.3f})")

print(f"End-effector position: ({joint_positions[-1][0]:.3f}, {joint_positions[-1][1]:.3f})")
```

## Inverse Kinematics

Inverse kinematics (IK) is the problem of determining the joint variables required to achieve a desired end-effector position and orientation. Unlike forward kinematics, IK can have multiple solutions, no solutions, or infinitely many solutions depending on the robot configuration and desired pose.

### Analytical Inverse Kinematics

For simple manipulators, analytical solutions can be found using geometric relationships.

```python
def inverse_kinematics_2link_planar(x, y, l1, l2):
    """
    Analytical solution for 2-link planar manipulator inverse kinematics.
    
    Args:
        x, y: Desired end-effector position
        l1, l2: Link lengths
    
    Returns:
        Tuple of two solutions (theta1_1, theta2_1) and (theta1_2, theta2_2)
        Returns None if no solution exists
    """
    # Calculate distance from origin to target
    r = np.sqrt(x**2 + y**2)
    
    # Check if target is reachable
    if r > l1 + l2:
        print("Target is outside reachable workspace")
        return None
    elif r < abs(l1 - l2):
        print("Target is inside inner workspace but unreachable")
        return None
    
    # Calculate theta2 using law of cosines
    cos_theta2 = (l1**2 + l2**2 - r**2) / (2 * l1 * l2)
    # Clamp to valid range to avoid numerical errors
    cos_theta2 = np.clip(cos_theta2, -1, 1)
    theta2 = np.arccos(cos_theta2)
    
    # Two possible solutions for theta2 (elbow up and elbow down)
    theta2_1 = theta2
    theta2_2 = -theta2
    
    # Calculate theta1 for each solution
    k1 = l1 + l2 * np.cos(theta2_1)
    k2 = l2 * np.sin(theta2_1)
    theta1_1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    
    k1 = l1 + l2 * np.cos(theta2_2)
    k2 = l2 * np.sin(theta2_2)
    theta1_2 = np.arctan2(y, x) - np.arctan2(k2, k1)
    
    return (theta1_1, theta2_1), (theta1_2, theta2_2)

# Example: Find joint angles to reach (1.0, 0.5) with 2-link arm
l1, l2 = 1.0, 0.8
target_pos = (1.0, 0.5)

solutions = inverse_kinematics_2link_planar(target_pos[0], target_pos[1], l1, l2)

if solutions:
    sol1, sol2 = solutions
    print(f"\nInverse Kinematics Solutions for target {target_pos}:")
    print("-" * 50)
    print(f"Solution 1: theta1 = {sol1[0]:.3f} rad ({np.degrees(sol1[0]):.1f}°), theta2 = {sol1[1]:.3f} rad ({np.degrees(sol1[1]):.1f}°)")
    print(f"Solution 2: theta1 = {sol2[0]:.3f} rad ({np.degrees(sol2[0]):.1f}°), theta2 = {sol2[1]:.3f} rad ({np.degrees(sol2[1]):.1f}°)")
    
    # Verify solutions by computing forward kinematics
    FK1 = forward_kinematics_planar([sol1[0], sol1[1]], [l1, l2])
    FK2 = forward_kinematics_planar([sol2[0], sol2[1]], [l1, l2])
    
    print(f"Verification - Solution 1 end-effector: ({FK1[-1][0]:.3f}, {FK1[-1][1]:.3f})")
    print(f"Verification - Solution 2 end-effector: ({FK2[-1][0]:.3f}, {FK2[-1][1]:.3f})")
else:
    print("No valid inverse kinematics solution found")
```

### Numerical Inverse Kinematics

For more complex robots or when analytical solutions are not feasible, numerical methods can be used.

```python
def jacobian_2link_planar(theta1, theta2, l1, l2):
    """
    Calculate the Jacobian matrix for a 2-link planar manipulator.
    
    Args:
        theta1, theta2: Joint angles
        l1, l2: Link lengths
    
    Returns:
        2x2 Jacobian matrix
    """
    jacobian = np.zeros((2, 2))
    jacobian[0, 0] = -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)  # ∂x/∂theta1
    jacobian[0, 1] = -l2 * np.sin(theta1 + theta2)                        # ∂x/∂theta2
    jacobian[1, 0] = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)    # ∂y/∂theta1
    jacobian[1, 1] = l2 * np.cos(theta1 + theta2)                         # ∂y/∂theta2
    
    return jacobian

def inverse_kinematics_numerical(target_pos, initial_joints, link_lengths, max_iterations=100, tolerance=1e-5):
    """
    Numerical inverse kinematics using the Jacobian transpose method.
    
    Args:
        target_pos: Desired end-effector position (x, y)
        initial_joints: Initial joint angles [theta1, theta2]
        link_lengths: Link lengths [l1, l2]
        max_iterations: Maximum number of iterations
        tolerance: Position tolerance for convergence
    
    Returns:
        Joint angles to reach target position, or None if failed to converge
    """
    # Current joint angles
    joints = np.array(initial_joints, dtype=float)
    l1, l2 = link_lengths
    
    for i in range(max_iterations):
        # Calculate current end-effector position using forward kinematics
        current_pos = forward_kinematics_planar(joints, link_lengths)[-1]
        
        # Calculate error
        error = np.array(target_pos) - np.array(current_pos)
        
        # Check if error is within tolerance
        if np.linalg.norm(error) < tolerance:
            print(f"Converged after {i+1} iterations")
            return joints
        
        # Calculate Jacobian
        J = jacobian_2link_planar(joints[0], joints[1], l1, l2)
        
        # Calculate joint angle change using Jacobian transpose method
        # dq = J^T * dx (simplified - in practice, you might use pseudoinverse)
        joint_change = J.T @ error * 0.1  # Learning rate factor
        
        # Update joint angles
        joints += joint_change
    
    print("Failed to converge within maximum iterations")
    return None

# Example: Use numerical method to find joint angles
initial_joints = [0.0, 0.0]  # Start with zero angles
numerical_solution = inverse_kinematics_numerical(target_pos, initial_joints, [l1, l2])

if numerical_solution is not None:
    print(f"\nNumerical IK Solution: theta1 = {numerical_solution[0]:.3f} rad, theta2 = {numerical_solution[1]:.3f} rad")
    
    # Verify solution
    final_pos = forward_kinematics_planar(numerical_solution, [l1, l2])[-1]
    print(f"Final position: ({final_pos[0]:.3f}, {final_pos[1]:.3f})")
    print(f"Target position: ({target_pos[0]}, {target_pos[1]})")
    print(f"Error: {np.linalg.norm(np.array(target_pos) - np.array(final_pos)):.6f}")
```

## Workspace Analysis

The workspace of a robot is the set of all points that the end-effector can reach. Understanding the workspace is important for robot design and task planning.

```python
def plot_workspace_2link(l1, l2, resolution=0.1):
    """
    Analyze and visualize the workspace of a 2-link planar manipulator.
    
    Args:
        l1, l2: Link lengths
        resolution: Resolution of the grid for analysis
    
    Returns:
        Tuple of (reachable_x, reachable_y) arrays
    """
    # Calculate workspace boundaries
    max_reach = l1 + l2
    min_reach = abs(l1 - l2)
    
    # Create a grid of points
    x_range = np.arange(-max_reach, max_reach, resolution)
    y_range = np.arange(-max_reach, max_reach, resolution)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Determine which points are reachable
    reachable = np.sqrt(X**2 + Y**2) <= max_reach
    unreachable = np.sqrt(X**2 + Y**2) < min_reach
    workspace = reachable & ~unreachable
    
    return X, Y, workspace

# Example: Analyze workspace of 2-link manipulator
X, Y, workspace = plot_workspace_2link(l1, l2)

# Calculate some statistics about the workspace
total_points = workspace.size
reachable_points = np.sum(workspace)
workspace_ratio = reachable_points / total_points

print(f"\nWorkspace Analysis for 2-link manipulator (l1={l1}, l2={l2}):")
print("-" * 60)
print(f"Total grid points analyzed: {total_points}")
print(f"Reachable grid points: {reachable_points}")
print(f"Workspace coverage: {workspace_ratio*100:.2f}%")
print(f"Maximum reach: {l1 + l2}")
print(f"Minimum reach: {abs(l1 - l2)}")

# Find boundary points
from scipy import ndimage
boundary = ndimage.binary_erosion(workspace) ^ workspace
boundary_coords = np.where(boundary)
boundary_x = X[boundary_coords]
boundary_y = Y[boundary_coords]

print(f"Workspace boundary points: {len(boundary_x)}")
```

## Implementation: Kinematic Chain Class

```python
class KinematicChain:
    """
    A class to represent and compute kinematics for a serial robot chain.
    """
    def __init__(self, dh_params_list):
        """
        Initialize the kinematic chain.
        
        Args:
            dh_params_list: List of DHParameters objects for each joint
        """
        self.dh_params = dh_params_list
        self.n_joints = len(dh_params_list)
    
    def forward_kinematics(self, joint_angles):
        """
        Calculate forward kinematics for the chain.
        
        Args:
            joint_angles: List of joint angles
            
        Returns:
            4x4 transformation matrix for end-effector pose
        """
        if len(joint_angles) != self.n_joints:
            raise ValueError(f"Expected {self.n_joints} joint angles, got {len(joint_angles)}")
        
        # Calculate transformation from base to end-effector
        T_total = np.eye(4)
        
        for i in range(self.n_joints):
            # Update the joint angle in DH parameters
            current_dh = DHParameters(
                self.dh_params[i].a,
                self.dh_params[i].alpha,
                self.dh_params[i].d,
                joint_angles[i]
            )
            T_joint = current_dh.to_transformation_matrix()
            T_total = T_total @ T_joint
        
        return T_total
    
    def jacobian(self, joint_angles):
        """
        Calculate the geometric Jacobian for the chain.
        
        Args:
            joint_angles: List of joint angles
            
        Returns:
            6xN Jacobian matrix (N = number of joints)
        """
        if len(joint_angles) != self.n_joints:
            raise ValueError(f"Expected {self.n_joints} joint angles, got {len(joint_angles)}")
        
        # Calculate all transformation matrices
        T_matrices = [np.eye(4)]  # Base frame
        current_T = np.eye(4)
        
        for i in range(self.n_joints):
            current_dh = DHParameters(
                self.dh_params[i].a,
                self.dh_params[i].alpha,
                self.dh_params[i].d,
                joint_angles[i]
            )
            T_joint = current_dh.to_transformation_matrix()
            current_T = current_T @ T_joint
            T_matrices.append(current_T)
        
        # Calculate end-effector position in base frame
        p_end = T_matrices[-1][0:3, 3]
        
        # Initialize Jacobian
        J = np.zeros((6, self.n_joints))
        
        # Calculate each column of the Jacobian
        for i in range(self.n_joints):
            z_i = T_matrices[i][0:3, 2]  # z-axis of joint i in base frame
            p_i = T_matrices[i][0:3, 3]  # origin of joint i in base frame
            
            # For revolute joints
            J[0:3, i] = np.cross(z_i, (p_end - p_i))  # Linear velocity contribution
            J[3:5, i] = z_i  # Angular velocity contribution
        
        return J
    
    def inverse_kinematics(self, target_pose, initial_joints, max_iterations=100, tolerance=1e-5):
        """
        Solve inverse kinematics using numerical methods.
        
        Args:
            target_pose: 4x4 target transformation matrix
            initial_joints: Initial joint angles guess
            max_iterations: Maximum number of iterations
            tolerance: Position/orientation tolerance for convergence
        
        Returns:
            Joint angles solution or None if failed
        """
        joints = np.array(initial_joints, dtype=float)
        
        for i in range(max_iterations):
            # Calculate current pose
            current_pose = self.forward_kinematics(joints)
            
            # Calculate error in position and orientation
            pos_error = target_pose[0:3, 3] - current_pose[0:3, 3]
            
            # For orientation error, use a simple approach (rotation vector)
            R_error = target_pose[0:3, 0:3] @ current_pose[0:3, 0:3].T
            # Get rotation angle from rotation matrix trace
            cos_angle = np.clip((np.trace(R_error) - 1) / 2, -1, 1)
            angle_error = np.arccos(cos_angle)
            
            # Calculate error magnitude
            error_magnitude = np.sqrt(np.sum(pos_error**2) + angle_error**2)
            
            if error_magnitude < tolerance:
                print(f"Converged after {i+1} iterations")
                return joints
            
            # Calculate Jacobian
            J = self.jacobian(joints)
            
            # Use pseudoinverse for more robust solution
            J_pinv = np.linalg.pinv(J)
            
            # Calculate desired change in joint space
            error_vector = np.concatenate([pos_error, np.zeros(3)])  # Simplified - just position
            joint_change = J_pinv @ error_vector * 0.1  # Learning rate
            
            # Update joints
            joints += joint_change
        
        print("Failed to converge within maximum iterations")
        return None

# Example: Create a simple 3-DOF manipulator and test kinematics
simple_dh = [
    DHParameters(0, -np.pi/2, 0.1, 0),      # Joint 1: Twist joint
    DHParameters(0.5, 0, 0, 0),             # Joint 2: Elbow joint
    DHParameters(0.4, 0, 0, 0)              # Joint 3: Wrist joint
]

robot = KinematicChain(simple_dh)

# Test forward kinematics
joint_config = [np.pi/4, -np.pi/6, np.pi/3]
end_pose = robot.forward_kinematics(joint_config)

print(f"\nTesting KinematicChain class:")
print("-" * 35)
print(f"Joint configuration: {[f'{ang:.3f}' for ang in joint_config]}")
print(f"End-effector position: ({end_pose[0,3]:.3f}, {end_pose[1,3]:.3f}, {end_pose[2,3]:.3f})")

# Calculate Jacobian
jac = robot.jacobian(joint_config)
print(f"Jacobian shape: {jac.shape}")
print(f"Position rows of Jacobian:\n{jac[0:3, :]}")
```

## Singularity Analysis

Singularities are configurations where the robot loses one or more degrees of freedom. Understanding and avoiding singularities is crucial for robot control.

```python
def find_singularities(robot_chain, joint_ranges, resolution=0.1):
    """
    Find singular configurations by analyzing the Jacobian determinant.
    
    Args:
        robot_chain: KinematicChain object
        joint_ranges: List of (min, max) for each joint
        resolution: Resolution for grid search
    
    Returns:
        List of singular joint configurations
    """
    singular_configs = []
    
    # For a 2-DOF example: simple grid search
    if robot_chain.n_joints == 2:
        joints1_range = np.arange(joint_ranges[0][0], joint_ranges[0][1], resolution)
        joints2_range = np.arange(joint_ranges[1][0], joint_ranges[1][1], resolution)
        
        for j1 in joints1_range:
            for j2 in joints2_range:
                joints = [j1, j2]
                J = robot_chain.jacobian(joints)
                J_pos = J[0:2, :]  # Position part of Jacobian for 2D case
                
                # Calculate determinant (near zero indicates singularity)
                det = np.linalg.det(J_pos)
                if abs(det) < 1e-3:  # Threshold for singularity
                    singular_configs.append(joints)
    
    return singular_configs

# Example: Find singularities for our 2-link manipulator
print(f"\nSingularity Analysis:")
print("-" * 20)
print("Analyzing singularities is complex for a general KinematicChain,")
print("but for our 2-link planar example, singularities occur when:")
print("1. θ2 = 0 (fully stretched arm)")
print("2. θ2 = π (fully folded arm)")
print("These correspond to the manipulator being fully extended or folded")

# For our 2-link example, singularity occurs when theta2 is 0 or π
joints_stretched = [0, 0]  # Arms fully stretched
joints_folded = [0, np.pi]  # Arms fully folded

J_stretched = jacobian_2link_planar(joints_stretched[0], joints_stretched[1], l1, l2)
J_folded = jacobian_2link_planar(joints_folded[0], joints_folded[1], l1, l2)

print(f"Jacobian determinant at stretched config: {np.linalg.det(J_stretched):.6f}")
print(f"Jacobian determinant at folded config: {np.linalg.det(J_folded):.6f}")
```

## Summary

This chapter covered the essential concepts of forward and inverse kinematics:

1. **Forward Kinematics**: Computing end-effector pose from joint angles using transformation matrices and the DH convention
2. **Inverse Kinematics**: Computing joint angles from desired end-effector pose using analytical and numerical methods
3. **Workspace Analysis**: Understanding the reachable workspace of robotic systems
4. **Jacobian Matrix**: Relating joint velocities to end-effector velocities
5. **Singularity Analysis**: Identifying configurations where the robot loses degrees of freedom

Kinematics forms the foundation of robot motion planning and control. Understanding these concepts is essential for controlling robots to perform various tasks effectively.

## Exercises

1. Implement the inverse kinematics solution for a 3-DOF planar manipulator using both analytical and numerical methods
2. Extend the KinematicChain class to handle prismatic joints in addition to revolute joints
3. Implement an algorithm to avoid singular configurations during trajectory execution
4. Create a visualization tool that shows the robot configuration in 3D space as joint angles change

## References

- Craig, J. J. (2005). *Introduction to Robotics: Mechanics and Control* (3rd ed.). Pearson.
- Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2006). *Robot Modeling and Control*. Wiley.
- Siciliano, B., & Khatib, O. (Eds.). (2016). *Springer Handbook of Robotics* (2nd ed.). Springer.