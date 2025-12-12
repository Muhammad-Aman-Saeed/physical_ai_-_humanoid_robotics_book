---
sidebar_position: 5
---

# Inverse Kinematics: Theory and Implementation

## Learning Objectives

After reading this chapter, you will be able to:
- Understand the mathematical foundations of inverse kinematics (IK)
- Implement analytical solutions for common robot configurations
- Apply numerical methods for solving inverse kinematics problems
- Identify and handle singularities in robotic systems
- Implement Jacobian-based methods for inverse kinematics
- Develop robust IK solvers with constraints handling

## Prerequisites

Before reading this chapter, you should:
- Understand forward kinematics and transformation matrices (Chapter 4)
- Be familiar with basic calculus and linear algebra
- Have knowledge of Jacobian matrices
- Basic programming skills in Python

## Introduction

Inverse kinematics is the process of determining the joint parameters needed to achieve a desired end-effector position and orientation. This is one of the most critical problems in robotics, as most robotic tasks require controlling the position and orientation of the robot's end-effector (e.g., gripper, tool, camera) rather than individual joint angles.

Unlike forward kinematics, which always has a unique solution, inverse kinematics can have multiple solutions, no solution, or infinitely many solutions depending on the robot configuration and desired pose. This makes IK both challenging and fascinating to solve.

## Types of Inverse Kinematics Solutions

### Closed-Form (Analytical) Solutions

For simple robots or special configurations, we can derive closed-form solutions to the inverse kinematics problem using geometric relationships and algebraic manipulations.

The advantages of analytical solutions include:
- Fast computation
- Direct insight into the kinematic structure
- No convergence issues

Let's examine a classical solution for a common configuration:

```python
import numpy as np

def inverse_kinematics_3dof_planar(x, y, l1, l2, l3):
    """
    Analytical solution for a 3-DOF planar manipulator.
    The first joint controls the rotation around the z-axis,
    while the other two joints operate in the x-y plane.
    """
    # First, solve for wrist position (before the last joint)
    # This approach works when the last joint is a wrist joint
    # that only controls end-effector orientation
    
    # For a 3R planar manipulator, we solve as if it's a 2R with
    # the first joint angle added to all others
    r = np.sqrt(x**2 + y**2)
    
    # Check reachability
    arm_length = l1 + l2 + l3
    if r > arm_length:
        print("Target position is beyond reach")
        return None
    elif r < abs(l1 - l2 - l3) or r < abs(l2 - l1 - l3) or r < abs(l3 - l1 - l2):
        print("Target position is too close to the base")
        return None
    
    # For a 2-link manipulator from previous chapter, extend to 3-link
    # This is a simplified example for educational purposes
    # Real 3DOF manipulator solutions would be more complex
    
    # Calculate joint 2 (theta2)
    cos_theta2 = (l1**2 + l2**2 - r**2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1, 1)  # Ensure valid range
    theta2 = np.arccos(cos_theta2)
    
    # Two possible solutions for theta2
    theta2_1 = theta2
    theta2_2 = -theta2
    
    # Calculate theta1 for each solution
    def calc_theta1(r, th2):
        k1 = l1 + l2 * np.cos(th2)
        k2 = l2 * np.sin(th2)
        return np.arctan2(y, x) - np.arctan2(k2, k1)
    
    theta1_1 = calc_theta1(r, theta2_1)
    theta1_2 = calc_theta1(r, theta2_2)
    
    # For a 3DOF robot, theta3 might control orientation at the end-effector
    # In this simplified example, we'll set it to achieve a desired orientation
    theta3_1 = 0  # Default orientation
    theta3_2 = 0  # Default orientation
    
    return [(theta1_1, theta2_1, theta3_1), (theta1_2, theta2_2, theta3_2)]

# Example usage
print("Analytical IK Example: 3-DOF Planar Manipulator")
print("-" * 50)
solutions = inverse_kinematics_3dof_planar(1.5, 0.5, 0.8, 0.7, 0.5)
if solutions:
    for i, sol in enumerate(solutions):
        print(f"Solution {i+1}: [θ1={sol[0]:.3f}, θ2={sol[1]:.3f}, θ3={sol[2]:.3f}]")
```

### Numerical Solutions

When analytical solutions are difficult or impossible to derive, numerical methods become essential. These methods iteratively converge to a solution.

#### Jacobian-Based Methods

The Jacobian matrix provides the relationship between joint velocities and end-effector velocities:

```
dx/dt = J(q) * dq/dt
```

Where:
- dx/dt is the end-effector velocity vector
- J(q) is the Jacobian matrix
- dq/dt is the joint velocity vector

Using this relationship, we can iteratively update joint angles:

```
Δq = J⁺ * Δx
```

Where J⁺ is the pseudoinverse of the Jacobian.

```python
def jacobian_transpose_3dof_planar(theta1, theta2, theta3, l1, l2, l3):
    """
    Calculate the Jacobian for a 3-DOF planar manipulator.
    This is a simplified model where the last joint only affects orientation.
    """
    # Positions of joints in the x-y plane
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    
    x3 = x2 + l3 * np.cos(theta1 + theta2 + theta3)
    y3 = y2 + l3 * np.sin(theta1 + theta2 + theta3)
    
    # Jacobian: J = [Jv, Jo] where Jv is velocity part, Jo is orientation part
    # For position (x, y), only the first 2 joints matter for planar motion
    J = np.zeros((3, 3))  # [dx, dy, dtheta] for 3 joints
    
    # Partial derivatives for x
    J[0, 0] = -l1*np.sin(theta1) - l2*np.sin(theta1 + theta2) - l3*np.sin(theta1 + theta2 + theta3)
    J[0, 1] = -l2*np.sin(theta1 + theta2) - l3*np.sin(theta1 + theta2 + theta3)
    J[0, 2] = -l3*np.sin(theta1 + theta2 + theta3)
    
    # Partial derivatives for y
    J[1, 0] = l1*np.cos(theta1) + l2*np.cos(theta1 + theta2) + l3*np.cos(theta1 + theta2 + theta3)
    J[1, 1] = l2*np.cos(theta1 + theta2) + l3*np.cos(theta1 + theta2 + theta3)
    J[1, 2] = l3*np.cos(theta1 + theta2 + theta3)
    
    # Partial derivatives for orientation (simplified as sum of joint angles)
    J[2, 0] = 1  # Full effect from first joint
    J[2, 1] = 1  # Full effect from second joint
    J[2, 2] = 1  # Full effect from third joint
    
    return J

def inverse_kinematics_jacobian(
    target_pos, 
    initial_joints, 
    link_lengths, 
    max_iterations=100, 
    tolerance=1e-5,
    step_size=0.1
):
    """
    Solve inverse kinematics using the Jacobian transpose method.
    
    Args:
        target_pos: Target position and orientation [x, y, theta]
        initial_joints: Initial joint angles [theta1, theta2, theta3]
        link_lengths: Link lengths [l1, l2, l3]
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        step_size: Step size for gradient descent
    
    Returns:
        Joint angles for target position or None if failed
    """
    joints = np.array(initial_joints, dtype=float)
    l1, l2, l3 = link_lengths
    
    for i in range(max_iterations):
        # Calculate current position
        current_angles = joints
        
        # Compute forward kinematics (simplified)
        x = l1 * np.cos(current_angles[0]) + l2 * np.cos(current_angles[0] + current_angles[1]) + \
            l3 * np.cos(current_angles[0] + current_angles[1] + current_angles[2])
        y = l1 * np.sin(current_angles[0]) + l2 * np.sin(current_angles[0] + current_angles[1]) + \
            l3 * np.sin(current_angles[0] + current_angles[1] + current_angles[2])
        theta = current_angles[0] + current_angles[1] + current_angles[2]
        
        current_pos = np.array([x, y, theta])
        
        # Calculate error
        error = np.array(target_pos) - current_pos
        
        # Check for convergence
        if np.linalg.norm(error) < tolerance:
            print(f"Converged after {i+1} iterations")
            return current_angles
        
        # Calculate Jacobian
        J = jacobian_transpose_3dof_planar(*current_angles, l1, l2, l3)
        
        # Calculate joint update using Jacobian transpose method
        # dq = J^T * dx
        joint_update = J.T @ error * step_size
        
        # Update joints
        joints = current_angles + joint_update
        
        # Optionally use pseudoinverse for better convergence
        # joint_update_pinv = np.linalg.pinv(J) @ error * step_size
        # joints = current_angles + joint_update_pinv
    
    print(f"Failed to converge after {max_iterations} iterations")
    return None

# Example: Calculate inverse kinematics for a 3-DOF manipulator
print("\nNumerical IK Example: 3-DOF Manipulator using Jacobian Method")
print("-" * 60)

target = [1.0, 0.8, 0.5]  # x, y, orientation
initial = [0.0, 0.0, 0.0]  # initial joint angles
links = [0.6, 0.5, 0.4]    # link lengths

solution = inverse_kinematics_jacobian(target, initial, links)

if solution is not None:
    print(f"Solution found: [θ1={solution[0]:.3f}, θ2={solution[1]:.3f}, θ3={solution[2]:.3f}]")
    
    # Verify the solution
    final_x = (links[0] * np.cos(solution[0]) + 
               links[1] * np.cos(solution[0] + solution[1]) + 
               links[2] * np.cos(solution[0] + solution[1] + solution[2]))
    final_y = (links[0] * np.sin(solution[0]) + 
               links[1] * np.sin(solution[0] + solution[1]) + 
               links[2] * np.sin(solution[0] + solution[1] + solution[2]))
    final_theta = solution[0] + solution[1] + solution[2]
    
    print(f"Verification: Actual pos [x={final_x:.3f}, y={final_y:.3f}, θ={final_theta:.3f}]")
    print(f"Target pos:   [x={target[0]:.3f}, y={target[1]:.3f}, θ={target[2]:.3f}]")
    print(f"Position error: {np.sqrt((final_x-target[0])**2 + (final_y-target[1])**2):.6f}")
```

### Jacobian Pseudoinverse Method

For better numerical stability and handling of redundant robots, the pseudoinverse method is preferred:

```python
def inverse_kinematics_pseudoinverse(
    target_pos,
    initial_joints,
    link_lengths,
    max_iterations=100,
    tolerance=1e-5,
    alpha=0.5  # Damping factor
):
    """
    Solve inverse kinematics using the damped least squares (DLS) method.
    
    Args:
        target_pos: Target position and orientation [x, y, theta]
        initial_joints: Initial joint angles [theta1, theta2, theta3]
        link_lengths: Link lengths [l1, l2, l3]
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        alpha: Damping factor for singularity handling
    
    Returns:
        Joint angles for target position or None if failed
    """
    joints = np.array(initial_joints, dtype=float)
    l1, l2, l3 = link_lengths
    
    for i in range(max_iterations):
        # Calculate current position using forward kinematics
        current_angles = joints
        
        # Position calculation
        x = l1 * np.cos(current_angles[0]) + l2 * np.cos(current_angles[0] + current_angles[1]) + \
            l3 * np.cos(current_angles[0] + current_angles[1] + current_angles[2])
        y = l1 * np.sin(current_angles[0]) + l2 * np.sin(current_angles[0] + current_angles[1]) + \
            l3 * np.sin(current_angles[0] + current_angles[1] + current_angles[2])
        theta = current_angles[0] + current_angles[1] + current_angles[2]
        
        current_pos = np.array([x, y, theta])
        
        # Calculate error
        error = np.array(target_pos) - current_pos
        
        # Check for convergence
        if np.linalg.norm(error) < tolerance:
            print(f"DLS converged after {i+1} iterations")
            return current_angles
        
        # Calculate Jacobian
        J = jacobian_transpose_3dof_planar(*current_angles, l1, l2, l3)
        
        # Calculate damped pseudoinverse: J# = J^T * (J * J^T + λ²I)^(-1)
        # Where λ (lambda) is related to alpha
        I = np.eye(J.shape[0])  # Identity matrix
        lambda_sq = alpha ** 2
        
        # Damped least squares approach
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_sq * I)
        
        # Calculate joint update
        joint_update = J_pinv @ error * 0.1  # Step size factor
        
        # Update joints
        joints = current_angles + joint_update
    
    print(f"DLS failed to converge after {max_iterations} iterations")
    return None

# Example: Compare Jacobian transpose vs pseudoinverse methods
print("\nComparison: Jacobian Transpose vs Pseudoinverse Methods")
print("-" * 55)

target_pos = [0.8, 1.0, 0.3]
initial_joints = [0.1, -0.2, 0.1]
links = [0.5, 0.5, 0.4]

# Try pseudoinverse method
solution_dls = inverse_kinematics_pseudoinverse(target_pos, initial_joints, links)

if solution_dls is not None:
    print(f"DLS Solution: [θ1={solution_dls[0]:.3f}, θ2={solution_dls[1]:.3f}, θ3={solution_dls[2]:.3f}]")
```

## Singularity Handling

Singularities occur when the Jacobian loses rank, meaning the robot loses one or more degrees of freedom. At singularities, the pseudoinverse becomes unstable.

```python
def detect_singularities(J, threshold=1e-3):
    """
    Detect if the Jacobian is near a singularity by examining its condition number.
    
    Args:
        J: Jacobian matrix
        threshold: Threshold for condition number (higher = more singular)
    
    Returns:
        True if near singularity, False otherwise
    """
    # Calculate condition number (ratio of largest to smallest singular values)
    cond_num = np.linalg.cond(J)
    return cond_num > 1/threshold

def damped_least_squares_with_singularity_handling(
    target_pos,
    initial_joints,
    link_lengths,
    max_iterations=100,
    tolerance=1e-5
):
    """
    Solve inverse kinematics with singularity handling using adaptive damping.
    """
    joints = np.array(initial_joints, dtype=float)
    l1, l2, l3 = link_lengths
    
    for i in range(max_iterations):
        # Forward kinematics
        current_angles = joints
        x = (l1 * np.cos(current_angles[0]) + 
             l2 * np.cos(current_angles[0] + current_angles[1]) + 
             l3 * np.cos(current_angles[0] + current_angles[1] + current_angles[2]))
        y = (l1 * np.sin(current_angles[0]) + 
             l2 * np.sin(current_angles[0] + current_angles[1]) + 
             l3 * np.sin(current_angles[0] + current_angles[1] + current_angles[2]))
        theta = current_angles[0] + current_angles[1] + current_angles[2]
        
        current_pos = np.array([x, y, theta])
        
        # Calculate error
        error = np.array(target_pos) - current_pos
        
        if np.linalg.norm(error) < tolerance:
            print(f"Adaptive DLS converged after {i+1} iterations")
            return current_angles
        
        # Calculate Jacobian
        J = jacobian_transpose_3dof_planar(*current_angles, l1, l2, l3)
        
        # Check for singularities and adjust damping
        is_singular = detect_singularities(J)
        if is_singular:
            # Increase damping near singularities
            alpha = 0.2  # Higher damping near singularity
        else:
            alpha = 0.01  # Lower damping when well-conditioned
        
        # Calculate damped pseudoinverse
        I = np.eye(J.shape[0])
        lambda_sq = alpha ** 2
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_sq * I)
        
        # Calculate joint update
        joint_update = J_pinv @ error * 0.1
        
        # Update joints
        joints = current_angles + joint_update
    
    print(f"Adaptive DLS failed to converge after {max_iterations} iterations")
    return None

# Example: Test singularity handling
print(f"\nSingularity Handling Example:")
print("-" * 30)

# Create a target at the maximum reach (possible singularity for 2-link)
# For 3-link, let's try a configuration where 2 links are collinear
singular_target = [l1 + l2 - 0.01, 0.0, 0.0]  # Near maximum reach
singular_initial = [0.0, 0.0, 0.0]

try_solution = damped_least_squares_with_singularity_handling(
    singular_target, 
    singular_initial, 
    links
)

if try_solution is not None:
    print(f"Singularity-handled solution: [θ1={try_solution[0]:.3f}, θ2={try_solution[1]:.3f}, θ3={try_solution[2]:.3f}]")
```

## Implementation: IK Solver Class

Let's create a comprehensive inverse kinematics solver class:

```python
class InverseKinematicsSolver:
    """
    A comprehensive Inverse Kinematics solver supporting multiple algorithms
    and constraints handling.
    """
    
    def __init__(self, dh_parameters, link_lengths):
        """
        Initialize the IK solver.
        
        Args:
            dh_parameters: DH parameters for the robot
            link_lengths: List of link lengths
        """
        self.dh_parameters = dh_parameters
        self.link_lengths = link_lengths
        self.n_joints = len(link_lengths)
    
    def forward_kinematics_planar(self, joint_angles):
        """
        Calculate forward kinematics for a planar manipulator.
        """
        if len(joint_angles) != self.n_joints:
            raise ValueError(f"Expected {self.n_joints} joint angles, got {len(joint_angles)}")
        
        x = 0
        y = 0
        cumulative_angle = 0
        
        for i, (angle, length) in enumerate(zip(joint_angles, self.link_lengths)):
            cumulative_angle += angle
            x += length * np.cos(cumulative_angle)
            y += length * np.sin(cumulative_angle)
        
        # Return position and total orientation
        return np.array([x, y, cumulative_angle])
    
    def jacobian_planar(self, joint_angles):
        """
        Calculate the Jacobian for a planar manipulator.
        """
        n = len(joint_angles)
        J = np.zeros((3, n))  # [dx, dy, dtheta] x n joints
        
        # Calculate cumulative position for each joint
        x_cum = 0
        y_cum = 0
        angle_cum = 0
        
        # Calculate end-effector position first
        final_pos = self.forward_kinematics_planar(joint_angles)
        xe, ye, _ = final_pos
        
        # For each joint i, calculate its contribution to end-effector velocity
        for i in range(n):
            # Position of joint i
            if i == 0:
                xi = 0
                yi = 0
                angle_to_i = joint_angles[0]
            else:
                temp_angles = joint_angles[:i]
                temp_lengths = self.link_lengths[:i]
                
                xj = 0
                yj = 0
                anglej = 0
                for j, (ang, length) in enumerate(zip(temp_angles, temp_lengths)):
                    anglej += ang
                    xj += length * np.cos(anglej)
                    yj += length * np.sin(anglej)
                
                xi = xj
                yi = yj
                angle_to_i = sum(joint_angles[:i+1])
            
            # Calculate partial derivatives
            # dx/dtheta_i = -Σ(l_j*sin(θ_sum)) for j >= i
            # dy/dtheta_i = Σ(l_j*cos(θ_sum)) for j >= i
            dx_dtheta = 0
            dy_dtheta = 0
            
            for j in range(i, n):
                angle_sum = sum(joint_angles[i:j+1])
                dx_dtheta -= self.link_lengths[j] * np.sin(angle_sum)
                dy_dtheta += self.link_lengths[j] * np.cos(angle_sum)
            
            dtheta_dtheta = 1  # Each joint contributes directly to final orientation
            
            J[0, i] = dx_dtheta  # ∂x/∂θ_i
            J[1, i] = dy_dtheta  # ∂y/∂θ_i
            J[2, i] = dtheta_dtheta  # ∂θ/∂θ_i
    
        return J
    
    def solve_analytical_2dof(self, target_x, target_y):
        """
        Analytical solution for a 2-DOF planar manipulator (if applicable).
        """
        if self.n_joints != 2:
            raise ValueError("This method only works for 2-DOF planar manipulators")
        
        l1, l2 = self.link_lengths
        
        # Check reachability
        r = np.sqrt(target_x**2 + target_y**2)
        max_reach = l1 + l2
        min_reach = abs(l1 - l2)
        
        if r > max_reach:
            return None  # Beyond reach
        elif r < min_reach:
            return None  # Inside inner reach
        
        # Calculate theta2
        cos_theta2 = (l1**2 + l2**2 - r**2) / (2 * l1 * l2)
        cos_theta2 = np.clip(cos_theta2, -1, 1)
        theta2 = np.arccos(cos_theta2)
        
        # Two solutions
        theta2_1 = theta2
        theta2_2 = -theta2
        
        # Calculate theta1 for each solution
        k1_1 = l1 + l2 * np.cos(theta2_1)
        k2_1 = l2 * np.sin(theta2_1)
        theta1_1 = np.arctan2(target_y, target_x) - np.arctan2(k2_1, k1_1)
        
        k1_2 = l1 + l2 * np.cos(theta2_2)
        k2_2 = l2 * np.sin(theta2_2)
        theta1_2 = np.arctan2(target_y, target_x) - np.arctan2(k2_2, k1_2)
        
        return [(theta1_1, theta2_1), (theta1_2, theta2_2)]
    
    def solve_jacobian_pseudoinverse(self, target_pos, initial_joints, 
                                    max_iterations=100, tolerance=1e-5, 
                                    alpha=0.01):
        """
        Solve IK using the damped least squares method.
        """
        joints = np.array(initial_joints, dtype=float)
        
        for i in range(max_iterations):
            # Calculate current position
            current_pos = self.forward_kinematics_planar(joints)
            
            # Calculate error
            error = np.array(target_pos) - current_pos
            
            # Check convergence
            if np.linalg.norm(error) < tolerance:
                print(f"Jacobian method converged in {i+1} iterations")
                return joints
            
            # Calculate Jacobian
            J = self.jacobian_planar(joints)
            
            # Calculate damped pseudoinverse
            I = np.eye(J.shape[0])
            J_pinv = J.T @ np.linalg.inv(J @ J.T + alpha**2 * I)
            
            # Update joints
            joints += J_pinv @ error * 0.1
        
        print(f"Jacobian method failed to converge after {max_iterations} iterations")
        return None
    
    def solve_with_constraints(self, target_pos, initial_joints, 
                              joint_limits=None, max_iterations=100):
        """
        Solve IK with joint limits constraints.
        """
        joints = np.array(initial_joints, dtype=float)
        original_joints = joints.copy()
        
        for i in range(max_iterations):
            # Calculate current position
            current_pos = self.forward_kinematics_planar(joints)
            
            # Calculate error
            error = np.array(target_pos) - current_pos
            
            # Check convergence
            if np.linalg.norm(error) < 1e-5:
                # Check if within joint limits
                if joint_limits:
                    for j, (joint, limits) in enumerate(zip(joints, joint_limits)):
                        if not (limits[0] <= joint <= limits[1]):
                            break
                    else:
                        print(f"Solution with constraints found in {i+1} iterations")
                        return joints
                else:
                    print(f"Solution with constraints found in {i+1} iterations")
                    return joints
            
            # Calculate Jacobian
            J = self.jacobian_planar(joints)
            
            # Calculate update
            J_pinv = np.linalg.pinv(J)
            joint_update = J_pinv @ error * 0.1
            
            # Apply update
            joints += joint_update
            
            # Apply joint limits if provided
            if joint_limits:
                for j in range(len(joints)):
                    joints[j] = np.clip(joints[j], joint_limits[j][0], joint_limits[j][1])
        
        print(f"Solution with constraints not found after {max_iterations} iterations")
        return None


# Example usage of the comprehensive IK solver
print("\nComprehensive IK Solver Example:")
print("-" * 35)

# Create a 3-DOF manipulator
ik_solver = InverseKinematicsSolver(None, [0.7, 0.6, 0.5])

# Test analytical solution for 2-DOF (will raise error as we have 3-DOF)
try:
    solutions_2dof = ik_solver.solve_analytical_2dof(0.8, 0.6)
except ValueError as e:
    print(f"Correctly caught error: {e}")

# Test numerical solution
target = [1.0, 0.5, 1.0]  # x, y, orientation
initial = [0.0, 0.0, 0.0]

solution = ik_solver.solve_jacobian_pseudoinverse(target, initial)

if solution is not None:
    print(f"Numerical solution: {[f'{angle:.3f}' for angle in solution]}")
    
    # Verify
    final_pos = ik_solver.forward_kinematics_planar(solution)
    print(f"Final pos: x={final_pos[0]:.3f}, y={final_pos[1]:.3f}, θ={final_pos[2]:.3f}")
    print(f"Target pos: x={target[0]:.3f}, y={target[1]:.3f}, θ={target[2]:.3f}")

# Test with joint limits
joint_limits = [(-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2)]
solution_limited = ik_solver.solve_with_constraints(target, initial, joint_limits)

if solution_limited is not None:
    print(f"Solution with limits: {[f'{angle:.3f}' for angle in solution_limited]}")
    print("All joint angles within limits:", all(
        limits[0] <= angle <= limits[1] 
        for angle, limits in zip(solution_limited, joint_limits)
    ))
```

## Jacobian Transpose and Pseudoinverse Comparison

Let's create a comparison of different methods:

```python
def compare_ik_methods(target_pos, initial_joints, link_lengths):
    """
    Compare different IK methods for the same target.
    """
    print(f"\nIK Methods Comparison for target {target_pos}")
    print("=" * 50)
    
    # Method 1: Jacobian Transpose
    print("Method 1: Jacobian Transpose")
    sol1 = inverse_kinematics_jacobian(target_pos, initial_joints, link_lengths, 
                                     max_iterations=100, step_size=0.1)
    if sol1 is not None:
        pos1 = [link_lengths[0]*np.cos(sol1[0]) + link_lengths[1]*np.cos(sol1[0]+sol1[1]) + 
                link_lengths[2]*np.cos(sol1[0]+sol1[1]+sol1[2]),
                link_lengths[0]*np.sin(sol1[0]) + link_lengths[1]*np.sin(sol1[0]+sol1[1]) + 
                link_lengths[2]*np.sin(sol1[0]+sol1[1]+sol1[2]),
                sol1[0]+sol1[1]+sol1[2]]
        err1 = np.linalg.norm(np.array(target_pos) - np.array(pos1))
        print(f"  Solution: {[f'{a:.3f}' for a in sol1]}")
        print(f"  Error: {err1:.6f}")
    else:
        print("  Failed to converge")
    
    # Method 2: Pseudoinverse
    print("\nMethod 2: Pseudoinverse (DLS)")
    sol2 = inverse_kinematics_pseudoinverse(target_pos, initial_joints, link_lengths)
    if sol2 is not None:
        pos2 = [link_lengths[0]*np.cos(sol2[0]) + link_lengths[1]*np.cos(sol2[0]+sol2[1]) + 
                link_lengths[2]*np.cos(sol2[0]+sol2[1]+sol2[2]),
                link_lengths[0]*np.sin(sol2[0]) + link_lengths[1]*np.sin(sol2[0]+sol2[1]) + 
                link_lengths[2]*np.sin(sol2[0]+sol2[1]+sol2[2]),
                sol2[0]+sol2[1]+sol2[2]]
        err2 = np.linalg.norm(np.array(target_pos) - np.array(pos2))
        print(f"  Solution: {[f'{a:.3f}' for a in sol2]}")
        print(f"  Error: {err2:.6f}")
    else:
        print("  Failed to converge")

# Compare methods for a reachable target
target = [1.2, 0.8, 0.5]
initial = [0.1, 0.1, 0.1]
links = [0.6, 0.5, 0.4]

compare_ik_methods(target, initial, links)
```

## Summary

This chapter covered inverse kinematics in depth:

1. **Analytical Solutions**: Direct mathematical formulas for simple robot configurations
2. **Numerical Solutions**: Iterative methods that work for complex robots using Jacobian matrices
3. **Jacobian-based Methods**: How joint velocities relate to end-effector velocities
4. **Singularity Handling**: Techniques for dealing with configurations where the robot loses DOF
5. **Implementation**: Practical approaches to solving IK with constraints

Inverse kinematics is essential for any robotic application where you need to control the end-effector position and orientation. The choice of method depends on the robot's complexity, computational requirements, and accuracy needs.

## Exercises

1. Implement the inverse kinematics solution for a 6-DOF robot arm using the Jacobian pseudoinverse method
2. Add orientation control to the numerical IK solver (currently only handles position)
3. Implement a hybrid approach that starts with an analytical solution and refines with numerical methods
4. Create a real-time IK solver that can handle multiple targets and generate smooth trajectories

## References

- Siciliano, B., & Khatib, O. (Eds.). (2016). *Springer Handbook of Robotics* (2nd ed.). Springer.
- Craig, J. J. (2005). *Introduction to Robotics: Mechanics and Control* (3rd ed.). Pearson.
- Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2006). *Robot Modeling and Control*. Wiley.