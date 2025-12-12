"""
Walking Controller Implementation Exercise
Chapter 6: Gait Planning and Locomotion

This exercise implements a walking controller to maintain balance during locomotion.
"""

import numpy as np
import matplotlib.pyplot as plt


class WalkingController:
    """
    A simple walking controller for biped robots implementing basic ZMP-based balance control.
    """
    def __init__(self, robot_height=0.85, step_length=0.3, step_width=0.2, gravity=9.81):
        """
        Initialize the walking controller.
        
        Args:
            robot_height: Height of the center of mass above ground (m)
            step_length: Length of each step (m)
            step_width: Width between feet (m)
            gravity: Acceleration due to gravity (m/s^2)
        """
        self.robot_height = robot_height
        self.step_length = step_length
        self.step_width = step_width
        self.gravity = gravity
        
        # Inverted pendulum model parameters
        self.omega = np.sqrt(gravity / robot_height)  # Natural frequency
        
        # Current state of the robot
        self.com_pos = np.array([0.0, 0.0, robot_height])  # CoM position [x, y, z]
        self.com_vel = np.array([0.0, 0.0, 0.0])          # CoM velocity [vx, vy, vz]
        
        # Foot positions
        self.left_foot_pos = np.array([0.0, step_width/2, 0.0])
        self.right_foot_pos = np.array([0.0, -step_width/2, 0.0])
        
        # Walking state
        self.current_support_foot = "left"  # "left", "right", or "both"
        self.step_phase = 0.0  # 0.0 to 1.0, progress through current step
        self.step_time = 0.8   # Time for each step (seconds)
        
        # Control parameters
        self.zmp_kp = 15.0  # Proportional gain for ZMP control
        self.zmp_kd = 2.0 * np.sqrt(self.zmp_kp)  # Derivative gain for critically damped system
        self.com_height_kp = 50.0  # Proportional gain for CoM height control
        self.com_height_kd = 2.0 * np.sqrt(self.com_height_kp)
        
        # Trajectories for plotting
        self.time_history = []
        self.com_trajectory = []
        self.zmp_trajectory = []
        self.foot_trajectory = []
        
        # ZMP tracking error accumulator
        self.total_zmp_error = 0.0
        self.zmp_error_samples = 0
        
    def calculate_zmp(self):
        """
        Calculate the current Zero Moment Point based on CoM position and estimated acceleration.
        
        Returns:
            (x, y) coordinates of ZMP
        """
        # Approximate CoM acceleration based on position error from desired ZMP
        # This is a simplified calculation; in practice, we'd have more accurate acceleration data
        x_com, y_com, z_com = self.com_pos
        v_x, v_y, v_z = self.com_vel
        
        # Estimate acceleration from the inverted pendulum model
        # ZMP_x = CoM_x - (h/g) * CoM_acc_x where h is CoM height
        # Rearranging: CoM_acc_x = (CoM_x - ZMP_x) * g/h
        # But since we're controlling ZMP, we'll estimate it differently
        
        # For this simulation, we estimate ZMP based on the support foot
        # and the desired walking direction
        if self.current_support_foot == "left":
            # ZMP is near the left foot
            estimated_zmp_x = self.left_foot_pos[0] - 0.02  # Slightly behind foot for stability
            estimated_zmp_y = self.left_foot_pos[1] + 0.01  # Slight inward positioning
        else:  # Right foot support
            estimated_zmp_x = self.right_foot_pos[0] - 0.02  # Slightly behind foot for stability
            estimated_zmp_y = self.right_foot_pos[1] - 0.01  # Slight inward positioning
            
        # Blend toward the next expected ZMP as we progress through the step
        next_support = "right" if self.current_support_foot == "left" else "left"
        if next_support == "right":
            next_zmp_x = self.right_foot_pos[0] - 0.02
            next_zmp_y = self.right_foot_pos[1] - 0.01
        else:
            next_zmp_x = self.left_foot_pos[0] - 0.02
            next_zmp_y = self.left_foot_pos[1] + 0.01
            
        # Advance ZMP toward next support as step progresses
        blend_factor = min(1.0, self.step_phase * 1.5)  # Move ZMP ahead of CoM
        estimated_zmp_x = estimated_zmp_x * (1 - blend_factor) + next_zmp_x * blend_factor
        estimated_zmp_y = estimated_zmp_y * (1 - blend_factor) + next_zmp_y * blend_factor
        
        return np.array([estimated_zmp_x, estimated_zmp_y])
    
    def update_desired_zmp(self):
        """
        Update the desired ZMP based on walking state and foot positions.
        """
        # For single support phase, ZMP should be near the support foot
        if self.current_support_foot == "left":
            # ZMP should be near left foot
            desired_x = self.left_foot_pos[0] - 0.02  # Slightly behind foot for stability
            desired_y = self.left_foot_pos[1] + 0.01  # Slight inward for balance
        elif self.current_support_foot == "right":
            # ZMP should be near right foot
            desired_x = self.right_foot_pos[0] - 0.02  # Slightly behind foot for stability
            desired_y = self.right_foot_pos[1] - 0.01  # Slight inward for balance
        else:
            # Double support phase - ZMP between feet
            desired_x = (self.left_foot_pos[0] + self.right_foot_pos[0]) / 2
            desired_y = (self.left_foot_pos[1] + self.right_foot_pos[1]) / 2
        
        # Add progressive movement toward next step
        # As we advance through the step, shift ZMP toward the next foot placement
        if self.current_support_foot == "left":
            # Moving toward where the right foot will be placed
            next_foot_x = self.right_foot_pos[0] + self.step_length
            next_foot_y = self.step_width / 2
        else:
            # Moving toward where the left foot will be placed
            next_foot_x = self.left_foot_pos[0] + self.step_length
            next_foot_y = -self.step_width / 2
            
        # Blend between current support and next target
        blend_factor = min(1.0, self.step_phase * 1.5)  # Advance ZMP ahead of CoM
        desired_x = desired_x * (1 - blend_factor) + next_foot_x * blend_factor
        desired_y = desired_y * (1 - blend_factor) + next_foot_y * blend_factor
        
        self.desired_zmp = np.array([desired_x, desired_y])
    
    def compute_balance_control(self, dt):
        """
        Compute balance control inputs using ZMP feedback.
        
        Args:
            dt: Time step (s)
        
        Returns:
            Control outputs dictionary
        """
        # Update desired ZMP based on walking state
        self.update_desired_zmp()
        
        # Calculate current ZMP
        current_zmp = self.calculate_zmp()
        
        # Calculate ZMP error
        zmp_error = self.desired_zmp - current_zmp
        
        # Store for statistics
        self.total_zmp_error += np.linalg.norm(zmp_error)
        self.zmp_error_samples += 1
        
        # Simple PD controller for ZMP tracking
        zmp_control_output = self.zmp_kp * zmp_error[:2]  # Take just x,y components
        
        # Calculate CoM position error relative to desired ZMP
        com_xy_error = np.array([self.com_pos[0], self.com_pos[1]]) - self.desired_zmp
        
        return {
            'zmp_control': zmp_control_output,
            'com_xy_error': com_xy_error,
            'zmp_error': zmp_error,
            'current_zmp': current_zmp,
            'desired_zmp': self.desired_zmp
        }
    
    def update_walking_state(self, dt):
        """
        Update the walking state based on control inputs.
        
        Args:
            dt: Time step (s)
        """
        # Update step phase
        self.step_phase += dt / self.step_time
        
        # Check if we need to switch support foot
        if self.step_phase >= 1.0:
            # Complete current step
            self.step_phase = 0.0
            
            # Switch support foot
            if self.current_support_foot == "left":
                self.current_support_foot = "right"
                # Move left foot forward to where right foot was plus step length
                self.left_foot_pos[0] = self.right_foot_pos[0] + self.step_length
                self.left_foot_pos[1] = self.step_width / 2
            else:  # Right was support foot
                self.current_support_foot = "left"
                # Move right foot forward to where left foot was plus step length
                self.right_foot_pos[0] = self.left_foot_pos[0] + self.step_length
                self.right_foot_pos[1] = -self.step_width / 2
        
        # Apply balance control
        control_outputs = self.compute_balance_control(dt)
        
        # Update CoM based on balance control
        # Using inverted pendulum dynamics approximation
        com_x, com_y, com_z = self.com_pos
        v_x, v_y, v_z = self.com_vel
        
        # Calculate desired accelerations to move CoM toward desired ZMP
        # Using inverted pendulum: double_dot_x = omega^2 * (x - zmp_x)
        desired_x_acc = self.omega**2 * (control_outputs['desired_zmp'][0] - com_x)
        desired_y_acc = self.omega**2 * (control_outputs['desired_zmp'][1] - com_y)
        
        # Apply gravity to maintain height
        z_acc = -self.gravity  # Gravity always acts downward
        
        # Apply control with some effectiveness (not 100% to allow natural dynamics)
        control_weight = 0.15  # How much of the control effort to apply
        x_acc = desired_x_acc * control_weight
        y_acc = desired_y_acc * control_weight
        z_acc = z_acc  # Gravity always applies
        
        # Update velocities
        self.com_vel[0] += x_acc * dt
        self.com_vel[1] += y_acc * dt
        self.com_vel[2] += z_acc * dt
        
        # Update positions
        self.com_pos[0] += self.com_vel[0] * dt
        self.com_pos[1] += self.com_vel[1] * dt
        self.com_pos[2] += self.com_vel[2] * dt
        
        # Prevent CoM from going below ground level
        self.com_pos[2] = max(self.robot_height * 0.95, self.com_pos[2])  # Allow slight variation but not fall
        
        # Store trajectory data
        self.time_history.append(len(self.time_history) * dt)  # Time tracking
        self.com_trajectory.append(self.com_pos.copy())
        self.zmp_trajectory.append(control_outputs['current_zmp'].copy())
        
        # Store support foot position for visualization
        support_pos = self.left_foot_pos if self.current_support_foot == "left" else self.right_foot_pos
        self.foot_trajectory.append(support_pos.copy())
        
        # Return control outputs for monitoring
        return control_outputs
    
    def get_statistics(self):
        """
        Get statistics about the walking performance.
        
        Returns:
            Dictionary with performance metrics
        """
        if self.zmp_error_samples == 0:
            avg_zmp_error = 0.0
        else:
            avg_zmp_error = self.total_zmp_error / self.zmp_error_samples
            
        return {
            'average_zmp_error': avg_zmp_error,
            'final_com_position': self.com_pos.copy(),
            'final_com_velocity': self.com_vel.copy(),
            'final_support_foot': self.current_support_foot
        }


def exercise_1_basic_balance_control():
    """
    Exercise 1: Implement basic balance control using ZMP feedback.
    """
    print("Exercise 1: Basic Balance Control with ZMP Feedback")
    print("=" * 55)
    
    # Create a walking controller
    controller = WalkingController(robot_height=0.8, step_length=0.3, step_width=0.2)
    
    print(f"Robot parameters: height={controller.robot_height}m, "
          f"step_length={controller.step_length}m, step_width={controller.step_width}m")
    print(f"Natural frequency: ω={controller.omega:.3f} rad/s, "
          f"period={2*np.pi/controller.omega:.3f}s")
    print()
    
    # Run simulation for a period
    dt = 0.02  # 50 Hz simulation
    duration = 4.0  # 4 seconds
    steps = int(duration / dt)
    
    for i in range(steps):
        control_output = controller.update_walking_state(dt)
        
        # Print status occasionally
        if i % int(1/dt) == 0:  # Every second
            time_elapsed = i * dt
            com_pos = controller.com_pos
            zmp_pos = control_output['current_zmp']
            error = np.linalg.norm(control_output['zmp_error'])
            print(f"t={time_elapsed:.2f}s: CoM=({com_pos[0]:.3f}, {com_pos[1]:.3f}, {com_pos[2]:.3f}), "
                  f"ZMP=({zmp_pos[0]:.3f}, {zmp_pos[1]:.3f}), error={error:.3f}m")
    
    # Print final statistics
    stats = controller.get_statistics()
    print(f"\nFinal position: CoM=({stats['final_com_position'][0]:.3f}, "
          f"{stats['final_com_position'][1]:.3f}, {stats['final_com_position'][2]:.3f})")
    print(f"Final velocity: ({stats['final_com_velocity'][0]:.3f}, "
          f"{stats['final_com_velocity'][1]:.3f}, {stats['final_com_velocity'][2]:.3f})")
    print(f"Average ZMP error: {stats['average_zmp_error']:.3f}m")
    print(f"Final support foot: {stats['final_support_foot']}")
    
    return controller


def exercise_2_adaptive_balance_control():
    """
    Exercise 2: Adaptive balance control that adjusts parameters based on disturbances.
    """
    print("\nExercise 2: Adaptive Balance Control")
    print("=" * 35)
    
    # Create a walking controller with different parameters
    controller = WalkingController(robot_height=0.85, step_length=0.3, step_width=0.15)
    
    print("Implementing adaptive balance control...")
    
    # Store initial parameters for potential adaptation
    initial_kp = controller.zmp_kp
    initial_kd = controller.zmp_kd
    
    dt = 0.02
    duration = 4.0
    steps = int(duration / dt)
    
    # Track some metrics for adaptation
    error_history = []
    adaptation_enabled = False  # For this example, we'll conditionally adapt
    
    for i in range(steps):
        # Apply a temporary disturbance at a specific time
        if 1.5 < i*dt < 1.6 and i*dt > 1.5:  # Apply push at 1.5 seconds
            # Add an external force to the CoM (simulating a push)
            controller.com_vel[0] += 0.15  # Push forward
            controller.com_vel[1] += 0.05  # Small lateral push
            print(f"At t={i*dt:.2f}s: Applied external disturbance (push)")
        
        control_output = controller.update_walking_state(dt)
        
        # Calculate current error to determine if adaptation is needed
        current_error = np.linalg.norm(control_output['zmp_error'])
        error_history.append(current_error)
        
        # Simple adaptation: if error is consistently high, increase gains
        if len(error_history) > 50:  # Need some history
            recent_avg_error = np.mean(error_history[-20:])
            if recent_avg_error > 0.05:  # If consistently off by more than 5cm
                # Increase control gains to respond more aggressively
                controller.zmp_kp = min(initial_kp * 2.0, 30.0)  # Don't increase too much
                controller.zmp_kd = min(initial_kd * 2.0, 10.0)
            else:
                # Use initial gains when error is low
                controller.zmp_kp = initial_kp
                controller.zmp_kd = initial_kd
        
        # Print status occasionally
        if i % int(1/dt) == 0:
            print(f"t={i*dt:.2f}s: CoM_x={controller.com_pos[0]:.3f}, "
                  f"Error={current_error:.3f}m, Kp={controller.zmp_kp:.2f}")
    
    # Print final statistics
    stats = controller.get_statistics()
    print(f"\nAdaptive control results:")
    print(f"Final position: CoM=({stats['final_com_position'][0]:.3f}, "
          f"{stats['final_com_position'][1]:.3f})")
    print(f"Average ZMP error: {stats['average_zmp_error']:.3f}m")
    
    return controller


def exercise_3_trajectory_generation():
    """
    Exercise 3: Generate walking trajectory with smooth transitions.
    """
    print("\nExercise 3: Smooth Walking Trajectory Generation")
    print("=" * 48)
    
    # Create a walking controller
    controller = WalkingController(robot_height=0.8, step_length=0.35, step_width=0.2)
    
    # Generate a simple walking pattern
    dt = 0.02
    duration = 6.0  # 6 seconds of walking
    steps = int(duration / dt)
    
    # Storage for trajectory visualization
    time_points = []
    com_x_trajectory = []
    com_y_trajectory = []
    zmp_x_trajectory = []
    zmp_y_trajectory = []
    support_foot_states = []
    
    # Simulate walking with trajectory tracking
    for i in range(steps):
        time_points.append(i * dt)
        
        control_output = controller.update_walking_state(dt)
        
        # Store trajectory data
        com_x_trajectory.append(controller.com_pos[0])
        com_y_trajectory.append(controller.com_pos[1])
        zmp_x_trajectory.append(control_output['current_zmp'][0])
        zmp_y_trajectory.append(control_output['current_zmp'][1])
        support_foot_states.append(1 if controller.current_support_foot == "left" else -1)
        
        # Print occasional updates
        if i % (int(1/dt) * 2) == 0:  # Every 2 seconds
            print(f"Time {i*dt:.1f}s: CoM at ({controller.com_pos[0]:.2f}, {controller.com_pos[1]:.2f}), "
                  f"ZMP at ({control_output['current_zmp'][0]:.2f}, {control_output['current_zmp'][1]:.2f})")
    
    # Plot the walking trajectory
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: XY trajectory of CoM and ZMP
    axes[0, 0].plot(com_x_trajectory, com_y_trajectory, 'b-', linewidth=2, label='CoM trajectory', alpha=0.8)
    axes[0, 0].plot(zmp_x_trajectory, zmp_y_trajectory, 'm-', linewidth=1, label='ZMP trajectory', alpha=0.8)
    
    # Mark important positions
    axes[0, 0].plot(controller.com_pos[0], controller.com_pos[1], 'ro', markersize=10, label='Final CoM')
    # Mark foot positions during the simulation
    left_foot_x = [p[0] for i, p in enumerate(controller.left_foot_trajectory) if support_foot_states[i] == -1]
    left_foot_y = [p[1] for i, p in enumerate(controller.left_foot_trajectory) if support_foot_states[i] == -1]
    right_foot_x = [p[0] for i, p in enumerate(controller.right_foot_trajectory) if support_foot_states[i] == 1]
    right_foot_y = [p[1] for i, p in enumerate(controller.right_foot_trajectory) if support_foot_states[i] == 1]
    
    axes[0, 0].set_xlabel('X Position (m)')
    axes[0, 0].set_ylabel('Y Position (m)')
    axes[0, 0].set_title('Walking Trajectory (X-Y Plane)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    axes[0, 0].legend()
    
    # Plot 2: X position over time
    axes[0, 1].plot(time_points, com_x_trajectory, 'b-', linewidth=2, label='CoM X', alpha=0.8)
    axes[0, 1].plot(time_points, zmp_x_trajectory, 'm--', linewidth=1, label='ZMP X', alpha=0.7)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('X Position (m)')
    axes[0, 1].set_title('X Position vs Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Y position over time
    axes[1, 0].plot(time_points, com_y_trajectory, 'b-', linewidth=2, label='CoM Y', alpha=0.8)
    axes[1, 0].plot(time_points, zmp_y_trajectory, 'm--', linewidth=1, label='ZMP Y', alpha=0.7)
    # Add support foot indicators
    support_changes = []
    prev_support = support_foot_states[0]
    for i, support in enumerate(support_foot_states):
        if support != prev_support:
            support_changes.append((time_points[i], support))
            prev_support = support
    for change_time, support in support_changes:
        foot_label = 'Left Foot' if support == 1 else 'Right Foot'
        axes[1, 0].axvline(change_time, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Y Position (m)')
    axes[1, 0].set_title('Y Position vs Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: ZMP tracking error over time
    zmp_errors = [np.linalg.norm(np.array([cx, cy]) - np.array([zx, zy])) 
                  for cx, cy, zx, zy in zip(com_x_trajectory, com_y_trajectory, zmp_x_trajectory, zmp_y_trajectory)]
    axes[1, 1].plot(time_points, zmp_errors, 'r-', linewidth=1, alpha=0.8)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('ZMP Tracking Error (m)')
    axes[1, 1].set_title('ZMP Tracking Error Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0.02, color='orange', linestyle='--', label='2cm tolerance')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    avg_speed = (com_x_trajectory[-1] - com_x_trajectory[0]) / duration
    avg_error = np.mean(zmp_errors)
    max_error = np.max(zmp_errors)
    print(f"\nTrajectory Statistics:")
    print(f"- Total distance traveled: {com_x_trajectory[-1] - com_x_trajectory[0]:.3f} meters")
    print(f"- Average forward speed: {avg_speed:.3f} m/s")
    print(f"- Average ZMP tracking error: {avg_error:.3f} meters")
    print(f"- Maximum ZMP tracking error: {max_error:.3f} meters")
    print(f"- Steps taken: approximately {len(support_changes)//2}")
    
    return controller


def main():
    """
    Run all exercises for the walking controller.
    """
    print("Walking Controller Implementation - Exercises")
    print("=" * 50)
    
    # Run all exercises
    controller1 = exercise_1_basic_balance_control()
    controller2 = exercise_2_adaptive_balance_control()
    controller3 = exercise_3_trajectory_generation()
    
    print("\nSummary of Key Concepts Learned:")
    print("1. Zero Moment Point (ZMP) is crucial for dynamic balance in bipedal walking")
    print("2. Feedback control helps maintain balance by adjusting CoM position")
    print("3. ZMP-based controllers can handle disturbances and adapt to changes")
    print("4. Smooth trajectory generation is important for natural-looking movement")
    print("5. The inverted pendulum model provides a good approximation for balance control")
    print()
    
    print("Exercise 1 Results:")
    stats1 = controller1.get_statistics()
    print(f"  Final position: ({stats1['final_com_position'][0]:.3f}, {stats1['final_com_position'][1]:.3f})")
    print(f"  Average ZMP error: {stats1['average_zmp_error']:.3f}m")
    
    print("Exercise 2 Results:")
    stats2 = controller2.get_statistics()
    print(f"  Final position: ({stats2['final_com_position'][0]:.3f}, {stats2['final_com_position'][1]:.3f})")
    print(f"  Average ZMP error: {stats2['average_zmp_error']:.3f}m")
    
    print("Exercise 3 Results:")
    print("  Trajectory generation completed with visualization")


if __name__ == "__main__":
    main()