"""
Forward Kinematics for Simple Arm Exercise
Chapter 4: Forward and Inverse Kinematics

This exercise implements forward kinematics for a simple 2D robotic arm.
"""

import numpy as np
import matplotlib.pyplot as plt


class Simple2DArm:
    """
    A class to represent a simple 2D robotic arm with 2 joints.
    """
    def __init__(self, link_lengths):
        """
        Initialize the robotic arm.
        
        Args:
            link_lengths: List of link lengths [L1, L2]
        """
        if len(link_lengths) != 2:
            raise ValueError("Simple2DArm expects exactly 2 link lengths")
        
        self.L1 = link_lengths[0]  # Length of first link
        self.L2 = link_lengths[1]  # Length of second link
    
    def forward_kinematics(self, theta1, theta2):
        """
        Calculate the end-effector position given joint angles.
        
        Args:
            theta1: Angle of first joint (in radians)
            theta2: Angle of second joint (in radians)
        
        Returns:
            (x, y) position of end-effector
        """
        # Calculate position of joint 2
        x1 = self.L1 * np.cos(theta1)
        y1 = self.L1 * np.sin(theta1)
        
        # Calculate position of end-effector (joint 3)
        x2 = x1 + self.L2 * np.cos(theta1 + theta2)
        y2 = y1 + self.L2 * np.sin(theta1 + theta2)
        
        return x2, y2
    
    def forward_kinematics_with_joints(self, theta1, theta2):
        """
        Calculate the end-effector position and intermediate joint positions.
        
        Args:
            theta1: Angle of first joint (in radians)
            theta2: Angle of second joint (in radians)
        
        Returns:
            Dictionary with joint positions:
            - 'base': (0, 0) - base position
            - 'joint1': (x1, y1) - position of joint 1
            - 'end_effector': (x2, y2) - position of end-effector
        """
        # Calculate positions
        x1 = self.L1 * np.cos(theta1)
        y1 = self.L1 * np.sin(theta1)
        
        x2 = x1 + self.L2 * np.cos(theta1 + theta2)
        y2 = y1 + self.L2 * np.sin(theta1 + theta2)
        
        return {
            'base': (0, 0),
            'joint1': (x1, y1),
            'end_effector': (x2, y2)
        }
    
    def draw_arm(self, theta1, theta2, title="Robotic Arm Configuration"):
        """
        Draw the robotic arm in its current configuration.
        
        Args:
            theta1: Angle of first joint (in radians)
            theta2: Angle of second joint (in radians)
            title: Title for the plot
        """
        positions = self.forward_kinematics_with_joints(theta1, theta2)
        
        # Extract joint positions
        base = positions['base']
        joint1 = positions['joint1']
        end_effector = positions['end_effector']
        
        # Create the plot
        plt.figure(figsize=(8, 8))
        
        # Draw links
        plt.plot([base[0], joint1[0]], [base[1], joint1[1]], 'b-', linewidth=4, label=f'Link 1 (L={self.L1})')
        plt.plot([joint1[0], end_effector[0]], [joint1[1], end_effector[1]], 'r-', linewidth=4, label=f'Link 2 (L={self.L2})')
        
        # Draw joints and end effector
        plt.plot(base[0], base[1], 'ko', markersize=10, label='Base')
        plt.plot(joint1[0], joint1[1], 'bo', markersize=10, label='Joint 1')
        plt.plot(end_effector[0], end_effector[1], 'ro', markersize=10, label='End Effector')
        
        # Mark the end-effector position
        plt.annotate(f'({end_effector[0]:.2f}, {end_effector[1]:.2f})', 
                     xy=end_effector, xytext=(10, 10), 
                     textcoords='offset points', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Set axis properties
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.xlim(-self.L1-self.L2-0.5, self.L1+self.L2+0.5)
        plt.ylim(-self.L1-self.L2-0.5, self.L1+self.L2+0.5)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.title(title)
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.legend()
        plt.show()


def exercise_1_basic_forward_kinematics():
    """
    Exercise 1: Basic Forward Kinematics Calculation
    """
    print("Exercise 1: Basic Forward Kinematics Calculation")
    print("=" * 50)
    
    # Create a simple arm with L1=1.0, L2=0.8
    arm = Simple2DArm([1.0, 0.8])
    
    # Test cases
    test_cases = [
        (0, 0),  # Both joints at 0 degrees
        (np.pi/2, 0),  # First joint 90 degrees, second at 0
        (0, np.pi/2),  # First joint 0, second joint 90 degrees
        (np.pi/4, np.pi/4),  # Both joints at 45 degrees
        (np.pi/3, -np.pi/6)  # First joint 60, second joint -30
    ]
    
    print("Calculating forward kinematics for various joint angles:")
    print("{:<15} {:<15} {:<20}".format("Theta1 (rad)", "Theta2 (rad)", "End-Effector (x, y)"))
    print("-" * 55)
    
    for theta1, theta2 in test_cases:
        x, y = arm.forward_kinematics(theta1, theta2)
        print(f"{theta1:>8.3f}      {theta2:>8.3f}      ({x:>6.3f}, {y:>6.3f})")


def exercise_2_arm_configuration_visualization():
    """
    Exercise 2: Visualizing Different Arm Configurations
    """
    print("\nExercise 2: Visualizing Different Arm Configurations")
    print("=" * 55)
    
    arm = Simple2DArm([1.0, 0.8])
    
    # Different configurations to visualize
    configurations = [
        (0, 0, "Both joints at 0 (fully extended horizontally)"),
        (np.pi/2, 0, "First joint at 90 deg (arm up)"),
        (0, np.pi/2, "Second joint at 90 deg (elbow bent up)"),
        (np.pi/4, -np.pi/4, "Elbow bent down"),
        (np.pi/3, np.pi/6, "Arbitrary configuration")
    ]
    
    for theta1, theta2, title in configurations[:2]:  # Just show first 2 to avoid too many plots
        print(f"Visualization: {title}")
        print(f"  Joint angles: Theta1={theta1:.3f} rad, Theta2={theta2:.3f} rad")
        x, y = arm.forward_kinematics(theta1, theta2)
        print(f"  End-effector position: ({x:.3f}, {y:.3f})")
        print("  Drawing arm configuration...")
        arm.draw_arm(theta1, theta2, title)


def exercise_3_workspace_analysis():
    """
    Exercise 3: Workspace Analysis
    """
    print("\nExercise 3: Workspace Analysis")
    print("=" * 35)
    
    arm = Simple2DArm([1.0, 0.8])
    
    # Calculate workspace boundaries
    max_reach = arm.L1 + arm.L2
    min_reach = abs(arm.L1 - arm.L2)
    
    print(f"Robot Arm Specifications:")
    print(f"  Link 1 length (L1): {arm.L1}")
    print(f"  Link 2 length (L2): {arm.L2}")
    print(f"  Maximum reach: {max_reach}")
    print(f"  Minimum reach (when L1 > L2): {min_reach}")
    print()
    
    print(f"Theoretical workspace:")
    print(f"  Maximum reach circle radius: {max_reach}")
    print(f"  Minimum reach circle radius: {min_reach}")
    print(f"  Workspace is annulus (ring-shaped) area between these radii")
    print()
    
    # Sample different joint angles to plot workspace
    theta1_vals = np.linspace(0, 2*np.pi, 50)
    theta2_vals = np.linspace(-np.pi, np.pi, 50)
    
    workspace_points = []
    for t1 in theta1_vals:
        for t2 in theta2_vals:
            x, y = arm.forward_kinematics(t1, t2)
            distance = np.sqrt(x**2 + y**2)
            # Only consider points within reachable range
            if min_reach <= distance <= max_reach:
                workspace_points.append((x, y))
    
    # Plot workspace
    if workspace_points:
        workspace_x, workspace_y = zip(*workspace_points)
        
        plt.figure(figsize=(10, 8))
        
        # Plot workspace points
        plt.scatter(workspace_x, workspace_y, s=1, alpha=0.5, label='Reachable workspace')
        
        # Draw circles for max and min reach
        angles = np.linspace(0, 2*np.pi, 100)
        max_circle_x = max_reach * np.cos(angles)
        max_circle_y = max_reach * np.sin(angles)
        min_circle_x = min_reach * np.cos(angles)
        min_circle_y = min_reach * np.sin(angles)
        
        plt.plot(max_circle_x, max_circle_y, 'r-', linewidth=2, label=f'Max reach ({max_reach})')
        if min_reach > 0:
            plt.plot(min_circle_x, min_circle_y, 'r-', linewidth=2, label=f'Min reach ({min_reach})')
        
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title('Robot Workspace Analysis')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.show()
        
        print(f"Plotted workspace with {len(workspace_points)} sample points")
        print(f"Approximate workspace area: {np.pi * (max_reach**2 - min_reach**2):.2f} square units")


def exercise_4_inverse_problem_verification():
    """
    Exercise 4: Verify Forward Kinematics with Known Inverse Solution
    """
    print("\nExercise 4: Verifying Forward Kinematics with Known Solutions")
    print("=" * 60)
    
    arm = Simple2DArm([1.0, 0.8])
    
    # For a 2-link arm, when the end-effector is at a specific position,
    # we can calculate the joint angles (inverse kinematics)
    # Let's verify using forward kinematics
    
    # Known solutions for specific positions
    # (x, y) -> (theta1, theta2) solutions
    test_positions_and_solutions = [
        # Easy cases
        ((arm.L1 + arm.L2, 0), (0, 0)),  # Fully stretched horizontally
        ((-(arm.L1 + arm.L2), 0), (np.pi, 0)),  # Fully stretched horizontally backward
        ((arm.L1 - arm.L2, 0), (0, np.pi)),  # Arm folded (if L1 > L2)
        ((0, arm.L1 + arm.L2), (np.pi/2, 0)),  # Arm up vertically
    ]
    
    print("Verifying forward kinematics with known inverse solutions:")
    print("{:<25} {:<30} {:<30}".format("Target Position", "Calculated Angles", "Forward Kin Result"))
    print("-" * 85)
    
    for target_pos, expected_angles in test_positions_and_solutions:
        if abs(target_pos[0]) > arm.L1 + arm.L2:  # Skip unreachable positions
            continue
            
        expected_theta1, expected_theta2 = expected_angles
        calculated_x, calculated_y = arm.forward_kinematics(expected_theta1, expected_theta2)
        
        print(f"({target_pos[0]:>6.2f}, {target_pos[1]:>6.2f})       "
              f"({expected_theta1:>6.3f}, {expected_theta2:>6.3f})         "
              f"({calculated_x:>6.3f}, {calculated_y:>6.3f})")


def main():
    """
    Run all exercises for forward kinematics.
    """
    print("Forward Kinematics for Simple Robotic Arm - Exercises")
    print("=" * 55)
    
    # Run all exercises
    exercise_1_basic_forward_kinematics()
    exercise_2_arm_configuration_visualization()
    exercise_3_workspace_analysis()
    exercise_4_inverse_problem_verification()
    
    print("\nSummary:")
    print("- Forward kinematics maps joint angles to end-effector position")
    print("- For a 2-link planar manipulator, the workspace is annulus-shaped")
    print("- The solution is always unique (unlike inverse kinematics)")
    print("- Understanding forward kinematics is essential before tackling inverse kinematics")


if __name__ == "__main__":
    main()