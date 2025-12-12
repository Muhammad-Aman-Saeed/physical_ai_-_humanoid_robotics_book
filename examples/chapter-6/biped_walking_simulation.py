"""
Biped walking simulation example
File: examples/chapter-6\biped_walking_simulation.py

This example demonstrates a simplified biped walking simulation using
a spring-loaded inverted pendulum (SLIP) model, which is commonly
used in humanoid robotics for gait planning and locomotion.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math


class BipedWalkingSimulator:
    """
    A simplified biped walking simulator using SLIP (Spring-Loaded Inverted Pendulum) model
    """
    
    def __init__(self):
        # Walking parameters
        self.step_length = 0.5  # meters
        self.step_height = 0.15  # meters
        self.step_duration = 1.0  # seconds
        self.step_frequency = 1.0 / self.step_duration
        
        # Robot parameters
        self.mass = 70  # kg
        self.gravity = 9.81  # m/s^2
        
        # Simulation parameters
        self.dt = 0.01  # time step
        self.ground_level = 0.0
        
        # Initialize state
        self.time = 0.0
        self.x_position = 0.0
        self.y_position = self.step_height  # initial height
        self.x_velocity = self.step_length / self.step_duration  # average velocity
        self.y_velocity = 0.0
        
        # Step parameters
        self.current_support_leg = "left"
        self.step_phase = 0.0  # 0.0 to 1.0, representing phase in current step
        
        # Data for visualization
        self.time_history = [self.time]
        self.x_history = [self.x_position]
        self.y_history = [self.y_position]
        self.support_leg_history = [self.current_support_leg]
    
    def compute_support_force(self, x, y):
        """
        Compute the ground reaction force based on the SLIP model
        """
        # For simplicity, we'll use a basic spring-damper model
        stiffness = 3000  # N/m
        damping = 200  # N*s/m
        
        # Ground contact force
        if y <= self.ground_level:
            y = self.ground_level  # prevent going below ground
            # Spring force when in contact with ground
            fy = stiffness * (self.step_height - y) - damping * self.y_velocity
            fx = 0  # For simplicity, assume no horizontal ground force
        else:
            # No ground contact
            fx, fy = 0, 0
        
        return fx, fy
    
    def update(self):
        """
        Update the simulation state for one time step
        """
        # Calculate forces
        support_fx, support_fy = self.compute_support_force(self.x_position, self.y_position)
        
        # Gravity force
        gravity_fy = -self.mass * self.gravity
        
        # Total forces
        total_fx = support_fx
        total_fy = support_fy + gravity_fy
        
        # Calculate accelerations (F = ma)
        ax = total_fx / self.mass
        ay = total_fy / self.mass
        
        # Update velocities
        self.x_velocity += ax * self.dt
        self.y_velocity += ay * self.dt
        
        # Update positions
        self.x_position += self.x_velocity * self.dt
        self.y_position += self.y_velocity * self.dt
        
        # Ensure robot doesn't go below ground
        if self.y_position < self.ground_level:
            self.y_position = self.ground_level
            self.y_velocity = max(0, self.y_velocity)  # Stop downward velocity
        
        # Update time
        self.time += self.dt
        
        # Update step phase
        self.step_phase += self.dt / self.step_duration
        if self.step_phase >= 1.0:
            self.step_phase = 0.0
            # Switch support leg
            self.current_support_leg = "right" if self.current_support_leg == "left" else "left"
            
            # Add some forward motion
            self.x_velocity = self.step_length / self.step_duration
        
        # Record history for plotting
        self.time_history.append(self.time)
        self.x_history.append(self.x_position)
        self.y_history.append(self.y_position)
        self.support_leg_history.append(self.current_support_leg)
    
    def simulate(self, duration=5.0):
        """
        Run the simulation for a given duration
        """
        num_steps = int(duration / self.dt)
        
        for _ in range(num_steps):
            self.update()
        
        return self.time_history, self.x_history, self.y_history


def plot_walking_trajectory(time_history, x_history, y_history):
    """
    Plot the walking trajectory
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot X vs Y (trajectory)
    ax1.plot(x_history, y_history, 'b-', linewidth=2, label='Robot trajectory')
    ax1.plot(x_history[0], y_history[0], 'go', markersize=10, label='Start')
    ax1.plot(x_history[-1], y_history[-1], 'ro', markersize=10, label='End')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Biped Walking Trajectory')
    ax1.grid(True)
    ax1.legend()
    
    # Plot Y position over time
    ax2.plot(time_history, y_history, 'r-', linewidth=2, label='Y Position')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Body Height Over Time')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def animate_walking(time_history, x_history, y_history):
    """
    Create an animation of the walking motion
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up the plot limits
    ax.set_xlim(min(x_history) - 0.5, max(x_history) + 0.5)
    ax.set_ylim(min(y_history) - 0.2, max(y_history) + 0.5)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Biped Walking Animation')
    ax.grid(True)
    
    # Draw ground
    ground_line = ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    
    # Robot representation (simplified as a circle with legs)
    robot_body = plt.Circle((0, 0), 0.1, color='blue', zorder=10)
    ax.add_artist(robot_body)
    
    # Animation function
    def animate(i):
        if i < len(x_history) and i < len(y_history):
            robot_body.center = (x_history[i], y_history[i])
            ax.set_title(f'Biped Walking Animation - Time: {time_history[i]:.2f}s')
        
        return [robot_body, ground_line]
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(x_history), interval=50, blit=True, repeat=True)
    plt.show()
    
    return anim


def main():
    print("Biped Walking Simulation using SLIP Model")
    print("=" * 50)
    
    # Create simulator
    simulator = BipedWalkingSimulator()
    
    print(f"Initial parameters:")
    print(f"  Step length: {simulator.step_length} m")
    print(f"  Step height: {simulator.step_height} m")
    print(f"  Step duration: {simulator.step_duration} s")
    print(f"  Step frequency: {simulator.step_frequency} Hz")
    print("")
    
    # Run simulation
    print("Running simulation for 5 seconds...")
    time_history, x_history, y_history = simulator.simulate(duration=5.0)
    
    print(f"Simulation completed with {len(time_history)} time steps")
    
    # Display final stats
    final_x = x_history[-1]
    final_y = y_history[-1]
    avg_velocity = final_x / time_history[-1]
    print(f"\nFinal stats:")
    print(f"  Final position: ({final_x:.2f}, {final_y:.2f}) m")
    print(f"  Average velocity: {avg_velocity:.2f} m/s")
    print(f"  Distance traveled: {final_x:.2f} m")
    
    # Create plots
    print("\nGenerating plots...")
    plot_walking_trajectory(time_history, x_history, y_history)
    
    print("\nGenerating animation...")
    anim = animate_walking(time_history, x_history, y_history)
    
    print("\nBiped walking simulation completed!")
    print("\nKey concepts illustrated:")
    print("- Spring-Loaded Inverted Pendulum (SLIP) model")
    print("- Center of mass trajectory during walking")
    print("- Ground contact and reaction forces")
    print("- Balance maintenance during locomotion")


if __name__ == "__main__":
    main()