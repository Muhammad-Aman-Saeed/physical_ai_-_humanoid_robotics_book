"""
Inverse Kinematics for 3-DOF Arm Exercise
Chapter 4: Forward and Inverse Kinematics

This exercise implements inverse kinematics for a 3-DOF planar robotic arm.
"""

import numpy as np
import matplotlib.pyplot as plt


class ThreeDOFPlanarArm:
    """
    A class to represent a 3-DOF planar robotic arm.
    """
    def __init__(self, link_lengths):
        """
        Initialize the 3-DOF planar robotic arm.
        
        Args:
            link_lengths: List of link lengths [L1, L2, L3]
        """
        if len(link_lengths) != 3:
            raise ValueError("ThreeDOFPlanarArm expects exactly 3 link lengths")
        
        self.L1 = link_lengths[0]  # Length of first link
        self.L2 = link_lengths[1]  # Length of second link  
        self.L3 = link_lengths[2]  # Length of third link
    
    def forward_kinematics(self, theta1, theta2, theta3):
        """
        Calculate the end-effector position given joint angles.
        
        Args:
            theta1: Angle of first joint (in radians)
            theta2: Angle of second joint (in radians)
            theta3: Angle of third joint (in radians)
        
        Returns:
            (x, y) position of end-effector
        """
        # Calculate position of joint 2
        x1 = self.L1 * np.cos(theta1)
        y1 = self.L1 * np.sin(theta1)
        
        # Calculate position of joint 3
        x2 = x1 + self.L2 * np.cos(theta1 + theta2)
        y2 = y1 + self.L2 * np.sin(theta1 + theta2)
        
        # Calculate position of end-effector (joint 4)
        x3 = x2 + self.L3 * np.cos(theta1 + theta2 + theta3)
        y3 = y2 + self.L3 * np.sin(theta1 + theta2 + theta3)
        
        return x3, y3
    
    def forward_kinematics_with_joints(self, theta1, theta2, theta3):
        """
        Calculate end-effector position and all joint positions.
        
        Args:
            theta1, theta2, theta3: Joint angles in radians
        
        Returns:
            Dictionary with all joint positions
        """
        # Calculate positions of all joints
        x1 = self.L1 * np.cos(theta1)
        y1 = self.L1 * np.sin(theta1)
        
        x2 = x1 + self.L2 * np.cos(theta1 + theta2)
        y2 = y1 + self.L2 * np.sin(theta1 + theta2)
        
        x3 = x2 + self.L3 * np.cos(theta1 + theta2 + theta3)
        y3 = y2 + self.L3 * np.sin(theta1 + theta2 + theta3)
        
        return {
            'base': (0, 0),
            'joint1': (x1, y1),
            'joint2': (x2, y2),
            'end_effector': (x3, y3)
        }
    
    def inverse_kinematics_2dof(self, x, y):
        """
        Analytical solution for the first 2 joints to reach (x, y) position.
        This reduces it to a 2-DOF problem by treating L2+L3 as a single link.
        
        Args:
            x, y: Target end-effector position
        
        Returns:
            List of joint angle solutions [(theta1_1, theta2_1), (theta1_2, theta2_2)]
            Returns None if no solution exists
        """
        # Calculate distance from origin to target
        r = np.sqrt(x**2 + y**2)
        
        # Effective second link length (treating L2+L3 as one link for 2DOF solution)
        L_effective = self.L2 + self.L3
        
        # Check if target is reachable
        if r > self.L1 + L_effective:
            print("Target is outside reachable workspace")
            return None
        elif r < abs(self.L1 - L_effective):
            print("Target is inside inner workspace but unreachable")
            return None
        
        # Calculate theta2 using law of cosines
        cos_theta2 = (self.L1**2 + L_effective**2 - r**2) / (2 * self.L1 * L_effective)
        cos_theta2 = np.clip(cos_theta2, -1, 1)  # Clamp to valid range to avoid numerical errors
        theta2 = np.arccos(cos_theta2)
        
        # Two possible solutions for theta2 (elbow up and elbow down)
        theta2_1 = theta2
        theta2_2 = -theta2
        
        # Calculate theta1 for each solution
        # For first solution
        k1_1 = self.L1 + L_effective * np.cos(theta2_1)
        k2_1 = L_effective * np.sin(theta2_1)
        theta1_1 = np.arctan2(y, x) - np.arctan2(k2_1, k1_1)
        
        # For second solution
        k1_2 = self.L1 + L_effective * np.cos(theta2_2)
        k2_2 = L_effective * np.sin(theta2_2)
        theta1_2 = np.arctan2(y, x) - np.arctan2(k2_2, k1_2)
        
        return [(theta1_1, theta2_1), (theta1_2, theta2_2)]
    
    def get_valid_workspace(self):
        """
        Calculate the valid workspace for this robot.
        
        Returns:
            (min_radius, max_radius) - workspace limits
        """
        max_reach = self.L1 + self.L2 + self.L3
        min_reach = max(0, self.L1 - self.L2 - self.L3)  # assuming L1 is longest
        return min_reach, max_reach


def exercise_1_analytical_ik_2dof():
    """
    Exercise 1: Analytical Inverse Kinematics for Reduced 2DOF Problem
    """
    print("Exercise 1: Analytical Inverse Kinematics for 2DOF Reduction")
    print("=" * 60)
    
    # Create a 3-DOF arm
    arm3dof = ThreeDOFPlanarArm([0.5, 0.4, 0.3])  # L1=0.5, L2=0.4, L3=0.3
    
    # Test with different target positions
    test_targets = [
        (0.8, 0.2),   # Reachable position
        (0.9, 0.0),   # Extended reach
        (0.2, 0.6),   # Upward position
    ]
    
    print(f"Robot specifications: L1={arm3dof.L1}, L2={arm3dof.L2}, L3={arm3dof.L3}")
    print(f"With L2+L3 effective length: {arm3dof.L2 + arm3dof.L3}")
    print()
    
    for x, y in test_targets:
        print(f"Target position: ({x}, {y})")
        solutions = arm3dof.inverse_kinematics_2dof(x, y)
        
        if solutions:
            for i, (theta1, theta2) in enumerate(solutions):
                print(f"  Solution {i+1}: (θ1={theta1:.3f}, θ2={theta2:.3f})")
                
                # Verify by computing forward kinematics
                # For this reduced model, we'll just apply to first 2 links
                x_check = arm3dof.L1 * np.cos(theta1) + (arm3dof.L2 + arm3dof.L3) * np.cos(theta1 + theta2)
                y_check = arm3dof.L1 * np.sin(theta1) + (arm3dof.L2 + arm3dof.L3) * np.sin(theta1 + theta2)
                
                print(f"    Verification: ({x_check:.3f}, {y_check:.3f}), Error: {np.sqrt((x-x_check)**2 + (y-y_check)**2):.6f}")
        else:
            print("  No solution found")
        print()


def exercise_2_detailed_inverse_kinematics():
    """
    Exercise 2: Detailed 3-DOF Inverse Kinematics Implementation
    """
    print("Exercise 2: Detailed 3-DOF Inverse Kinematics Implementation")
    print("=" * 55)
    
    # This will demonstrate the complexity of 3-DOF inverse kinematics
    # In a full implementation, this would include orientation control
    
    print("For a 3-DOF planar manipulator, we have 3 joints but only 2 task variables (x, y).")
    print("This creates a redundancy, with infinitely many solutions for most positions.")
    print("One common approach is to set one joint value (e.g., θ3) and solve for the other two.")
    print()
    
    # Example: Set theta3 and solve for theta1 and theta2 using a 2-DOF solution
    def inverse_kinematics_3dof_with_fixed_theta3(arm, x, y, theta3):
        """
        Solve IK for 3-DOF arm with fixed theta3.
        Move the origin to the third joint position and solve as 2-DOF problem.
        """
        # Calculate where joint 3 needs to be to reach target with given theta3
        # The end-effector is at the third joint plus the contribution from the third link
        # So joint 3 should be at: target - L3*[cos(theta1+theta2+theta3), sin(theta1+theta2+theta3)]
        
        # This is a complex nonlinear equation, so we'll use numerical methods
        # For demonstration, we'll implement a simplified version
        
        print(f"Solving for 3-DOF with fixed θ3 = {theta3:.3f} rad ({np.degrees(theta3):.1f}°)")
        print(f"Target position: ({x:.3f}, {y:.3f})")
        
        # In a real implementation, this would use numerical optimization methods
        # like the Jacobian-based methods we'll see in Exercise 4
        print("In a complete implementation, this would use numerical methods like:")
        print("- Jacobian transpose method")
        print("- Jacobian pseudoinverse method")
        print("- Damped Least Squares (DLS) method")
        print("- Cyclic Coordinate Descent (CCD)")
    
    # Set up a sample arm and target
    arm = ThreeDOFPlanarArm([0.6, 0.5, 0.3])
    target_x, target_y = 0.9, 0.4
    fixed_theta3 = np.pi/6  # 30 degrees
    
    inverse_kinematics_3dof_with_fixed_theta3(arm, target_x, target_y, fixed_theta3)
    
    
def jacobian_3dof_planar(theta1, theta2, theta3, l1, l2, l3):
    """
    Calculate the Jacobian for a 3-DOF planar manipulator.
    This is the matrix of partial derivatives of end-effector position wrt joint angles.
    
    Args:
        theta1, theta2, theta3: Joint angles
        l1, l2, l3: Link lengths
    
    Returns:
        2x3 Jacobian matrix (2 task vars x 3 joint vars)
    """
    # For a planar manipulator, we have 2 task variables (x, y) and 3 joint vars
    # The Jacobian will be 2x3
    J = np.zeros((2, 3))
    
    # x = l1*cos(θ1) + l2*cos(θ1+θ2) + l3*cos(θ1+θ2+θ3)
    # y = l1*sin(θ1) + l2*sin(θ1+θ2) + l3*sin(θ1+θ2+θ3)
    
    # dx/dθ1 = -l1*sin(θ1) - l2*sin(θ1+θ2) - l3*sin(θ1+θ2+θ3)
    J[0, 0] = -l1*np.sin(theta1) - l2*np.sin(theta1 + theta2) - l3*np.sin(theta1 + theta2 + theta3)
    
    # dx/dθ2 = -l2*sin(θ1+θ2) - l3*sin(θ1+θ2+θ3)
    J[0, 1] = -l2*np.sin(theta1 + theta2) - l3*np.sin(theta1 + theta2 + theta3)
    
    # dx/dθ3 = -l3*sin(θ1+θ2+θ3)
    J[0, 2] = -l3*np.sin(theta1 + theta2 + theta3)
    
    # dy/dθ1 = l1*cos(θ1) + l2*cos(θ1+θ2) + l3*cos(θ1+θ2+θ3)
    J[1, 0] = l1*np.cos(theta1) + l2*np.cos(theta1 + theta2) + l3*np.cos(theta1 + theta2 + theta3)
    
    # dy/dθ2 = l2*cos(θ1+θ2) + l3*cos(θ1+θ2+θ3)
    J[1, 1] = l2*np.cos(theta1 + theta2) + l3*np.cos(theta1 + theta2 + theta3)
    
    # dy/dθ3 = l3*cos(θ1+θ2+θ3)
    J[1, 2] = l3*np.cos(theta1 + theta2 + theta3)
    
    return J


def exercise_3_jacobian_computation():
    """
    Exercise 3: Jacobian Computation and Properties
    """
    print("\nExercise 3: Jacobian Computation and Properties")
    print("=" * 50)
    
    # Create an example robot
    arm = ThreeDOFPlanarArm([0.5, 0.4, 0.3])
    
    # Test configuration
    theta1, theta2, theta3 = np.pi/4, np.pi/6, -np.pi/6
    
    print(f"Robot configuration: θ1={theta1:.3f}, θ2={theta2:.3f}, θ3={theta3:.3f}")
    print(f"Robot parameters: L1={arm.L1}, L2={arm.L2}, L3={arm.L3}")
    print()
    
    # Compute Jacobian
    J = jacobian_3dof_planar(theta1, theta2, theta3, arm.L1, arm.L2, arm.L3)
    print(f"Jacobian matrix (2x3):")
    print(J)
    print()
    
    # Calculate forward kinematics for reference
    x, y = arm.forward_kinematics(theta1, theta2, theta3)
    print(f"End-effector position: ({x:.3f}, {y:.3f})")
    print()
    
    # Interpretation of Jacobian columns
    print("Interpretation of Jacobian columns:")
    print("  Column 0 (dθ1=1): Change in end-effector position per unit change in θ1")
    print("  Column 1 (dθ2=1): Change in end-effector position per unit change in θ2")
    print("  Column 2 (dθ3=1): Change in end-effector position per unit change in θ3")
    print()
    
    # Note: Since this is a redundant system (more joints than task variables),
    # the Jacobian is not square, so we can't compute its determinant directly.
    # Instead, we can look at properties of J*J^T
    JJT = J @ J.T
    print(f"J*J^T matrix (2x2):")
    print(JJT)
    print()
    
    # Eigenvalues of J*J^T tell us about the manipulability
    eigenvals = np.linalg.eigvals(JJT)
    print(f"Eigenvalues of J*J^T: {eigenvals}")
    print(f"Condition number: {np.sqrt(np.max(eigenvals)/np.min(eigenvals)):.3f}")
    print("(Lower condition number = better conditioning, less susceptibility to singularities)")


def numerical_inverse_kinematics_3dof(target_pos, initial_joints, arm, max_iterations=100, tolerance=1e-5):
    """
    Solve inverse kinematics numerically using the Jacobian transpose method.
    
    Args:
        target_pos: Desired end-effector position (x, y)
        initial_joints: Initial joint angles guess [theta1, theta2, theta3]
        arm: ThreeDOFPlanarArm object
        max_iterations: Maximum number of iterations
        tolerance: Position tolerance for convergence
    
    Returns:
        Joint angles solution or None if failed to converge
    """
    joints = np.array(initial_joints, dtype=float)
    target_x, target_y = target_pos
    
    for i in range(max_iterations):
        # Calculate current end-effector position
        current_x, current_y = arm.forward_kinematics(*joints)
        current_pos = np.array([current_x, current_y])
        
        # Calculate error
        error = np.array([target_x, target_y]) - current_pos
        
        # Check for convergence
        if np.linalg.norm(error) < tolerance:
            print(f"Converged after {i+1} iterations")
            return joints
        
        # Calculate Jacobian
        J = jacobian_3dof_planar(joints[0], joints[1], joints[2], arm.L1, arm.L2, arm.L3)
        
        # Use Jacobian transpose method: dq = J^T * dx
        # Since the system is redundant, this will find a solution that minimizes joint movement
        joint_change = J.T @ error * 0.1  # Learning rate factor
        
        # Apply update
        joints += joint_change
        
        # Limit the change to prevent wild movements
        max_change = 0.5
        if np.linalg.norm(joint_change) > max_change:
            joints -= joint_change
            joints += joint_change * max_change / np.linalg.norm(joint_change)
    
    print(f"Failed to converge after {max_iterations} iterations")
    return None


def exercise_4_numerical_inverse_kinematics():
    """
    Exercise 4: Numerical Inverse Kinematics Methods
    """
    print("\nExercise 4: Numerical Inverse Kinematics Methods")
    print("=" * 50)
    
    # Create a 3-DOF arm
    arm = ThreeDOFPlanarArm([0.5, 0.4, 0.3])
    
    # Define a target position
    target_pos = (0.7, 0.5)
    initial_guess = [0.1, 0.1, 0.1]  # Small initial angles
    
    print(f"Target position: ({target_pos[0]:.3f}, {target_pos[1]:.3f})")
    print(f"Initial guess: θ1={initial_guess[0]:.3f}, θ2={initial_guess[1]:.3f}, θ3={initial_guess[2]:.3f}")
    print(f"Robot params: L1={arm.L1}, L2={arm.L2}, L3={arm.L3}")
    print()
    
    # Solve using numerical method
    solution = numerical_inverse_kinematics_3dof(target_pos, initial_guess, arm)
    
    if solution is not None:
        print(f"Numerical solution found: θ1={solution[0]:.3f}, θ2={solution[1]:.3f}, θ3={solution[2]:.3f}")
        
        # Verify solution
        x_verify, y_verify = arm.forward_kinematics(*solution)
        error = np.sqrt((x_verify - target_pos[0])**2 + (y_verify - target_pos[1])**2)
        
        print(f"Verification: FK gives ({x_verify:.3f}, {y_verify:.3f})")
        print(f"Error: {error:.6f}")
        
        # Show how small changes in joints affect position
        print("\nSensitivity analysis:")
        J_sensitivity = jacobian_3dof_planar(solution[0], solution[1], solution[2], arm.L1, arm.L2, arm.L3)
        print(f"Jacobian at solution:\n{J_sensitivity}")
    else:
        print("No solution found using numerical method")


def exercise_5_workspace_analysis():
    """
    Exercise 5: Workspace Analysis for 3-DOF Arm
    """
    print("\nExercise 5: Workspace Analysis for 3-DOF Arm")
    print("=" * 45)
    
    arm = ThreeDOFPlanarArm([0.6, 0.5, 0.3])
    
    min_r, max_r = arm.get_valid_workspace()
    print(f"Robot specifications: L1={arm.L1}, L2={arm.L2}, L3={arm.L3}")
    print(f"Outer workspace radius: {max_r}")
    print(f"Inner workspace radius: {min_r}")
    print(f"Workspace is an annulus (ring-shaped) with area: {np.pi * (max_r**2 - min_r**2):.3f}")
    print()
    
    # Sample the workspace to visualize it
    n_samples = 1000
    theta1_vals = np.random.uniform(-np.pi, np.pi, n_samples)
    theta2_vals = np.random.uniform(-np.pi, np.pi, n_samples)
    theta3_vals = np.random.uniform(-np.pi, np.pi, n_samples)
    
    workspace_points = []
    for t1, t2, t3 in zip(theta1_vals, theta2_vals, theta3_vals):
        x, y = arm.forward_kinematics(t1, t2, t3)
        r = np.sqrt(x**2 + y**2)
        # Only include points that are within theoretical workspace
        if min_r - 0.01 <= r <= max_r + 0.01:  # Small tolerance for numerical errors
            workspace_points.append((x, y))
    
    if workspace_points:
        x_vals, y_vals = zip(*workspace_points)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(x_vals, y_vals, s=1, alpha=0.5)
        
        # Draw theoretical workspace boundaries
        theta = np.linspace(0, 2*np.pi, 100)
        inner_x = min_r * np.cos(theta)
        inner_y = min_r * np.sin(theta)
        outer_x = max_r * np.cos(theta)
        outer_y = max_r * np.sin(theta)
        
        plt.plot(outer_x, outer_y, 'r-', linewidth=2, label=f'Outer boundary (r={max_r:.2f})')
        if min_r > 0:
            plt.plot(inner_x, inner_y, 'r-', linewidth=2, label=f'Inner boundary (r={min_r:.2f})')
        
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title('3-DOF Planar Manipulator Workspace')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.show()
        
        print(f"Sampled {len(workspace_points)} valid points out of {n_samples} attempted")
        print(f"Visualization shows the reachable workspace of the 3-DOF manipulator")


def main():
    """
    Run all exercises for inverse kinematics.
    """
    print("Inverse Kinematics for 3-DOF Planar Robotic Arm - Exercises")
    print("=" * 65)
    
    # Run all exercises
    exercise_1_analytical_ik_2dof()
    exercise_2_detailed_inverse_kinematics()
    exercise_3_jacobian_computation()
    exercise_4_numerical_inverse_kinematics()
    exercise_5_workspace_analysis()
    
    print("\nSummary:")
    print("- 3-DOF planar manipulator is redundant (more joints than task variables)")
    print("- Multiple solutions exist for the same end-effector position")
    print("- Numerical methods like Jacobian-based approaches work well for redundant systems")
    print("- The Jacobian provides important information about local manipulability")
    print("- Workspace analysis is essential for understanding robot capabilities")


if __name__ == "__main__":
    main()