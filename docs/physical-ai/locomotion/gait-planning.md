---
sidebar_position: 6
---

# Gait Planning and Locomotion

## Learning Objectives

After reading this chapter, you will be able to:
- Understand the principles of bipedal and quadrupedal locomotion
- Analyze different walking patterns and their stability characteristics
- Implement basic gait planning algorithms for humanoids
- Understand the Zero Moment Point (ZMP) and its role in stable walking
- Create simple walking controllers that maintain balance
- Implement trajectory planning for stepping motions

## Prerequisites

Before reading this chapter, you should:
- Understand basic kinematics and dynamics (Chapters 2, 4)
- Be familiar with control theory basics
- Have knowledge of coordinate systems and transformations

## Introduction

Locomotion is the ability to move from one place to another. For humanoid robots, achieving stable, efficient, and human-like locomotion is one of the most challenging problems in robotics. Unlike wheeled or tracked robots, legged robots must manage complex dynamics, maintain balance during movement, and handle impacts with the ground.

Humanoid gait planning involves carefully orchestrating the movement of legs to achieve forward motion while maintaining the robot's balance. This requires understanding of dynamics, control theory, and biomechanics.

## Principles of Bipedal Locomotion

The fundamental challenge in bipedal locomotion is maintaining the center of mass (CoM) within the support polygon defined by the feet. When the CoM moves outside this polygon, the robot will fall.

Two main types of walking exist:
1. **Static walking**: The CoM remains within the support polygon at all times
2. **Dynamic walking**: The CoM is allowed to move outside the support polygon, requiring continuous balancing

### Center of Mass and Stability

```python
import numpy as np
import matplotlib.pyplot as plt

class CenterOfMassCalculator:
    """
    Calculator for center of mass in a multi-link system.
    """
    def __init__(self, masses, positions):
        """
        Initialize with masses and positions of segments.
        
        Args:
            masses: List of masses for each segment
            positions: List of (x, y, z) positions for each segment
        """
        self.masses = np.array(masses)
        self.positions = np.array(positions)
    
    def calculate_com(self):
        """
        Calculate the center of mass for the system.
        
        Returns:
            (x, y, z) coordinates of center of mass
        """
        total_mass = np.sum(self.masses)
        if total_mass == 0:
            return np.array([0.0, 0.0, 0.0])
        
        weighted_positions = self.masses[:, np.newaxis] * self.positions
        com = np.sum(weighted_positions, axis=0) / total_mass
        return com
    
    def calculate_com_trajectory(self, trajectories):
        """
        Calculate CoM trajectory over time.
        
        Args:
            trajectories: List of position trajectories for each segment
                         Each trajectory is a list of (x, y, z) positions over time
        
        Returns:
            List of CoM positions over time
        """
        com_trajectory = []
        n_steps = len(trajectories[0])
        
        for t in range(n_steps):
            # Get positions at time t for all segments
            positions_at_t = [traj[t] for traj in trajectories]
            self.positions = np.array(positions_at_t)
            com_at_t = self.calculate_com()
            com_trajectory.append(com_at_t)
        
        return com_trajectory

# Example: Simple two-mass system (like a simplified walking model)
masses = [5.0, 5.0]  # Two equal masses
positions = [[0.0, 0.5, 0.0], [0.5, 0.2, 0.0]]  # Initial positions

com_calc = CenterOfMassCalculator(masses, positions)
com_pos = com_calc.calculate_com()

print(f"Center of mass for two-mass system: ({com_pos[0]:.3f}, {com_pos[1]:.3f}, {com_pos[2]:.3f})")

# Example: Simulate CoM motion during walking
def simulate_double_support_phase():
    """
    Simulate a simplified double support phase of walking.
    """
    # Masses (simplified humanoid model)
    masses = [10.0, 10.0, 20.0, 5.0, 5.0]  # Head, arms, torso, left leg, right leg
    
    # Simulate movement of a simple walking step
    n_steps = 50
    trajectories = [[] for _ in range(len(masses))]
    
    # Simplified walking motion
    for t in np.linspace(0, 2*np.pi, n_steps):
        # Head moves forward and up/down
        head_pos = [0.3 + 0.5*t/(2*np.pi), 0.7 + 0.05*np.sin(t), 0.0]
        
        # Arms swing opposite to legs
        arm_offset = 0.1 * np.sin(t)
        left_arm = [head_pos[0] - 0.3, head_pos[1] + arm_offset, 0.1]
        right_arm = [head_pos[0] + 0.3, head_pos[1] - arm_offset, -0.1]
        
        # Torso follows head but with less up/down motion
        torso_pos = [head_pos[0], head_pos[1] - 0.1, 0.0]
        
        # Legs move in a walking pattern
        left_leg = [0.2 + 0.2*np.sin(t), 0.1, 0.0]  # Swinging leg
        right_leg = [0.3 - 0.2*np.sin(t), 0.1, 0.0]  # Stance leg
        
        positions_at_t = [head_pos, left_arm, right_arm, torso_pos, left_leg, right_leg][:len(masses)]
        
        # Store for each segment
        for i in range(len(masses)):
            trajectories[i].append(positions_at_t[i])
    
    # Calculate CoM trajectory
    com_trajectory_calc = CenterOfMassCalculator(masses, trajectories[0])
    # We'll calculate frame by frame
    com_trajectory = []
    for t in range(n_steps):
        frame_positions = [traj[t] for traj in trajectories]
        com_trajectory_calc.positions = np.array(frame_positions)
        com_at_t = com_trajectory_calc.calculate_com()
        com_trajectory.append(com_at_t)
    
    return com_trajectory, trajectories

com_trajectory, segment_trajectories = simulate_double_support_phase()
print(f"\nSimulated CoM trajectory for walking - first 5 positions:")
for i in range(min(5, len(com_trajectory))):
    print(f"  Step {i+1}: ({com_trajectory[i][0]:.3f}, {com_trajectory[i][1]:.3f}, {com_trajectory[i][2]:.3f})")
```

## Zero Moment Point (ZMP) Theory

The Zero Moment Point is a critical concept in walking robots. It represents the point on the ground where the net moment due to gravity and inertial forces is zero.

```python
def calculate_zmp(com_pos, com_acceleration, gravity=9.81, z_height=0.8):
    """
    Calculate the Zero Moment Point based on Center of Mass position and acceleration.
    
    Args:
        com_pos: (x, y, z) position of center of mass
        com_acceleration: (ax, ay, az) acceleration of center of mass
        gravity: Gravitational acceleration (default 9.81 m/s²)
        z_height: Height of CoM above ground (default 0.8 m)
    
    Returns:
        (x, y) coordinates of ZMP
    """
    x_com, y_com, z_com = com_pos
    x_acc, y_acc, z_acc = com_acceleration
    
    # ZMP equations (simplified for horizontal movement)
    # ZMP_x = x_com - z_height * x_acc / (gravity + z_acc)
    # ZMP_y = y_com - z_height * y_acc / (gravity + z_acc)
    
    zmp_x = x_com - z_height * x_acc / (gravity + z_acc)
    zmp_y = y_com - z_height * y_acc / (gravity + z_acc)
    
    return zmp_x, zmp_y

def simulate_zmp_during_walking():
    """
    Simulate ZMP during a walking step.
    """
    n_steps = 100
    zmp_trajectory = []
    
    # Simulate CoM motion during walking
    for t in np.linspace(0, 2*np.pi, n_steps):
        # Simulated CoM position (moving forward with slight up/down)
        com_x = 0.5 + 0.6*t/(2*np.pi)  # Moving forward
        com_y = 0.0 + 0.05*np.sin(2*t)  # Small lateral movement
        com_z = 0.8 + 0.02*np.sin(4*t)  # Small vertical movement
        
        # Simulated CoM acceleration
        com_x_acc = 0.6*np.cos(t)/(2*np.pi) * 0.5  # Forward acceleration
        com_y_acc = 0.05*2*np.cos(2*t) * 0.5      # Lateral acceleration  
        com_z_acc = -0.02*4*np.sin(4*t) * 0.5     # Vertical acceleration
        
        com_pos = (com_x, com_y, com_z)
        com_acc = (com_x_acc, com_y_acc, com_z_acc)
        
        zmp = calculate_zmp(com_pos, com_acc)
        zmp_trajectory.append(zmp)
    
    return zmp_trajectory

zmp_trajectory = simulate_zmp_during_walking()
print(f"\nZMP trajectory - first 5 points:")
for i in range(min(5, len(zmp_trajectory))):
    print(f"  Step {i+1}: ({zmp_trajectory[i][0]:.3f}, {zmp_trajectory[i][1]:.3f})")
```

## Inverted Pendulum Model

The linear inverted pendulum model (LIPM) is a common simplification for walking control:

```python
class InvertedPendulumModel:
    """
    Linear Inverted Pendulum Model for walking control.
    """
    def __init__(self, height, gravity=9.81):
        """
        Initialize the LIPM.
        
        Args:
            height: Height of the pendulum (CoM height)
            gravity: Gravitational acceleration
        """
        self.height = height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / height)  # Natural frequency
    
    def compute_zmp_from_com(self, com_pos, com_vel):
        """
        Compute ZMP given CoM position and velocity.
        
        Args:
            com_pos: (x, y) position of center of mass
            com_vel: (vx, vy) velocity of center of mass
        
        Returns:
            (x, y) coordinates of ZMP
        """
        x_com, y_com = com_pos
        x_vel, y_vel = com_vel
        
        # ZMP = CoM - (CoM_velocity / omega^2)
        zmp_x = x_com - x_vel / (self.omega**2)
        zmp_y = y_com - y_vel / (self.omega**2)
        
        return zmp_x, zmp_y
    
    def compute_com_from_zmp(self, zmp_pos, com_pos, dt=0.01):
        """
        Update CoM position based on ZMP reference.
        
        Args:
            zmp_pos: (x, y) desired ZMP position
            com_pos: (x, y) current CoM position
            dt: Time step
        
        Returns:
            Updated (x, y) CoM position
        """
        x_zmp, y_zmp = zmp_pos
        x_com, y_com = com_pos
        
        # Compute desired CoM acceleration based on ZMP
        x_acc = self.omega**2 * (x_com - x_zmp)
        y_acc = self.omega**2 * (y_com - y_zmp)
        
        # Update CoM position using simple integration
        # (in practice, you'd use the current velocity too)
        x_new = x_com + dt * self.omega**2 * (x_com - x_zmp) * dt
        y_new = y_com + dt * self.omega**2 * (y_com - y_zmp) * dt
        
        return x_new, y_new

# Example: Simulate LIPM for walking
def simulate_lipm_walking():
    """
    Simulate walking using the Linear Inverted Pendulum Model.
    """
    lipm = InvertedPendulumModel(height=0.85)
    
    # Initial conditions
    current_com = (0.1, 0.0)  # Start slightly forward
    current_vel = (0.0, 0.0)
    
    zmp_trajectory = []
    com_trajectory = []
    
    # Simulate walking for several steps
    for step in range(100):
        # Define ZMP trajectory for walking (alternating feet)
        # Simplified: ZMP moves to where the next foot will be
        step_half = step % 50
        if step < 50:  # Left foot stance phase
            desired_zmp_x = 0.1 + 0.001 * step  # Move ZMP forward gradually
            desired_zmp_y = 0.05  # Left foot offset
        else:  # Right foot stance phase
            desired_zmp_x = 0.1 + 0.001 * step
            desired_zmp_y = -0.05  # Right foot offset
        
        desired_zmp = (desired_zmp_x, desired_zmp_y)
        zmp_trajectory.append(desired_zmp)
        com_trajectory.append(current_com)
        
        # Update CoM based on ZMP
        # This is a simplified forward simulation
        x_com, y_com = current_com
        x_vel, y_vel = current_vel
        
        # Update velocity (acceleration based on ZMP error)
        x_acc = lipm.omega**2 * (x_com - desired_zmp_x)
        y_acc = lipm.omega**2 * (y_com - desired_zmp_y)
        
        # Update position and velocity (simple Euler integration)
        dt = 0.01
        x_vel_new = x_vel + x_acc * dt
        y_vel_new = y_vel + y_acc * dt
        
        x_com_new = x_com + x_vel_new * dt  
        y_com_new = y_com + y_vel_new * dt
        
        current_com = (x_com_new, y_com_new)
        current_vel = (x_vel_new, y_vel_new)
    
    return com_trajectory, zmp_trajectory

com_lipm, zmp_lipm = simulate_lipm_walking()
print(f"\nLIPM simulation - first 5 CoM positions:")
for i in range(min(5, len(com_lipm))):
    print(f"  Step {i+1}: CoM=({com_lipm[i][0]:.3f}, {com_lipm[i][1]:.3f}), ZMP=({zmp_lipm[i][0]:.3f}, {zmp_lipm[i][1]:.3f})")
```

## Walking Pattern Generation

### Footstep Planning

```python
class FootstepPlanner:
    """
    Plan footsteps for humanoid walking.
    """
    def __init__(self, step_length=0.3, step_width=0.2, step_height=0.05):
        """
        Initialize the footstep planner.
        
        Args:
            step_length: Forward distance per step
            step_width: Lateral distance between feet
            step_height: Maximum height of swinging foot
        """
        self.step_length = step_length
        self.step_width = step_width
        self.step_height = step_height
    
    def plan_footsteps(self, distance, start_position=(0, 0), start_orientation=0):
        """
        Plan a sequence of footsteps to cover a given distance.
        
        Args:
            distance: Total distance to travel
            start_position: Starting (x, y) position
            start_orientation: Starting orientation in radians
        
        Returns:
            List of (x, y, theta) for each footstep
        """
        footsteps = []
        
        # Calculate number of steps needed
        n_steps = int(distance / self.step_length) + 1
        
        x, y = start_position
        orientation = start_orientation
        
        # Add starting stance
        footsteps.append((x, y, orientation))  # Left foot
        
        for i in range(1, n_steps + 1):
            # Alternate between left and right foot
            if i % 2 == 1:  # Right foot step
                # Move forward
                x += self.step_length * np.cos(orientation)
                y += self.step_length * np.sin(orientation) - self.step_width
            else:  # Left foot step
                # Move forward
                x += self.step_length * np.cos(orientation)
                y += self.step_length * np.sin(orientation) + self.step_width
            
            # Add the footstep
            footsteps.append((x, y, orientation))
        
        return footsteps
    
    def generate_foot_trajectory(self, start_pos, end_pos, step_height=None, n_points=20):
        """
        Generate a smooth trajectory for a foot to move from start to end position.
        
        Args:
            start_pos: Starting (x, y, z) position
            end_pos: Ending (x, y, z) position  
            step_height: Maximum step height (if None, uses default)
            n_points: Number of points in trajectory
        
        Returns:
            List of (x, y, z) positions for the foot trajectory
        """
        if step_height is None:
            step_height = self.step_height
            
        trajectory = []
        
        start_x, start_y, start_z = start_pos
        end_x, end_y, end_z = end_pos
        
        # Interpolate position linearly
        for i in range(n_points):
            t = i / (n_points - 1)  # Progress from 0 to 1
            
            # Linear interpolation for x, y, z
            x = start_x + t * (end_x - start_x)
            y = start_y + t * (end_y - start_y)
            
            # For z, create an arc to lift the foot
            # Use a parabolic arc to raise the foot in the middle of the step
            z = start_z + t * (end_z - start_z)
            
            # Add arc for foot lifting (parabolic)
            if t < 0.5:
                # First half of step: lift foot
                z += step_height * 4 * t * (0.5 - t)
            else:
                # Second half of step: lower foot
                z += step_height * 4 * (1 - t) * (t - 0.5)
            
            trajectory.append((x, y, z))
        
        return trajectory

# Example: Plan footsteps and generate trajectories
footstep_planner = FootstepPlanner()

# Plan footsteps for 1 meter forward
footsteps = footstep_planner.plan_footsteps(1.0, start_position=(0, 0))
print("\nPlanned footsteps:")
for i, (x, y, theta) in enumerate(footsteps[:6]):  # Show first 6 steps
    print(f"  Step {i}: ({x:.3f}, {y:.3f}, θ={theta:.3f})")

# Generate trajectory for a single step
start_pos = (0.0, 0.1, 0.0)  # Left foot position
end_pos = (0.3, -0.1, 0.0)   # Right foot position after step
foot_trajectory = footstep_planner.generate_foot_trajectory(start_pos, end_pos)

print(f"\nFoot trajectory (first 5 points):")
for i in range(min(5, len(foot_trajectory))):
    x, y, z = foot_trajectory[i]
    print(f"  Point {i}: ({x:.3f}, {y:.3f}, {z:.3f})")
```

## Walking Controller Implementation

```python
class WalkingController:
    """
    A simple walking controller implementing basic balance and gait control.
    """
    def __init__(self, robot_height=0.8, gravity=9.81):
        # Robot parameters
        self.robot_height = robot_height
        self.gravity = gravity
        
        # Walking parameters
        self.step_length = 0.3
        self.step_width = 0.2
        self.step_height = 0.05
        self.step_time = 0.5  # Time for each step
        
        # ZMP controller parameters
        self.zmp_kp = 10.0
        self.zmp_kd = 2.0 * np.sqrt(self.zmp_kp)  # Critically damped
        
        # Current state
        self.current_com = np.array([0.0, 0.0, robot_height])
        self.current_com_vel = np.array([0.0, 0.0, 0.0])
        self.current_zmp = np.array([0.0, 0.0])
        self.support_foot = "left"  # Which foot is currently supporting weight
        
        # Step phase: 0.0 to 1.0, represents progress through current step
        self.step_phase = 0.0
        
        # For ZMP calculation
        self.omega = np.sqrt(gravity / robot_height)
    
    def update_reference_zmp(self, com_pos, com_vel, desired_zmp):
        """
        Update the reference ZMP based on current state and desired ZMP.
        """
        # Calculate current actual ZMP
        x_com, y_com, z_com = com_pos
        x_vel, y_vel, z_vel = com_vel
        
        current_zmp_x = x_com - self.robot_height * x_vel / (self.gravity + z_vel)
        current_zmp_y = y_com - self.robot_height * y_vel / (self.gravity + z_vel)
        
        actual_zmp = np.array([current_zmp_x, current_zmp_y])
        
        # Calculate error
        error = desired_zmp - actual_zmp
        
        # Simple PD control on ZMP error
        zmp_correction = self.zmp_kp * error  # + self.zmp_kd * error_derivative (if available)
        
        # Return corrected ZMP
        return actual_zmp + 0.1 * zmp_correction  # Use only part of correction for stability
    
    def compute_next_com_state(self, current_com, current_vel, target_zmp, dt):
        """
        Compute the next CoM state based on target ZMP.
        """
        x_com, y_com, z_com = current_com
        x_vel, y_vel, z_vel = current_vel
        
        # Based on inverted pendulum model: double support phase
        # Calculate acceleration to move CoM toward target ZMP
        x_acc = self.omega**2 * (x_com - target_zmp[0])
        y_acc = self.omega**2 * (y_com - target_zmp[1])
        z_acc = -self.gravity  # Always downward acceleration due to gravity
        
        # Update velocities
        new_x_vel = x_vel + x_acc * dt
        new_y_vel = y_vel + y_acc * dt  
        new_z_vel = z_vel + z_acc * dt
        
        # Update positions
        new_x_com = x_com + new_x_vel * dt
        new_y_com = y_com + new_y_vel * dt
        new_z_com = z_com + new_z_vel * dt
        
        # Limit Z position to be above ground
        new_z_com = max(self.robot_height, new_z_com)  # Don't let CoM go below robot height
        
        return np.array([new_x_com, new_y_com, new_z_com]), np.array([new_x_vel, new_y_vel, new_z_vel])
    
    def update_walking_step(self, dt):
        """
        Update the walking state for a single time step.
        """
        # Update step phase
        self.step_phase += dt / self.step_time
        
        if self.step_phase >= 1.0:
            # Complete current step, start next step
            self.step_phase = 0.0
            # Switch support foot
            self.support_foot = "right" if self.support_foot == "left" else "left"
        
        # Determine desired ZMP based on step phase
        if self.support_foot == "left":
            # Left foot is support foot, ZMP should be around left foot
            desired_zmp = np.array([-0.05, 0.1])  # Some position near left foot
        else:
            # Right foot is support foot, ZMP should be around right foot  
            desired_zmp = np.array([-0.05, -0.1])  # Some position near right foot
        
        # Add forward progression to ZMP to move robot forward
        forward_progress = self.step_length * 0.1  # Progress toward next foot placement
        desired_zmp[0] += forward_progress * self.step_phase  # Progress as step goes on
        
        # Update ZMP reference based on current state
        adjusted_zmp = self.update_reference_zmp(self.current_com, self.current_com_vel, desired_zmp)
        
        # Compute next CoM state
        self.current_com, self.current_com_vel = self.compute_next_com_state(
            self.current_com, self.current_com_vel, adjusted_zmp, dt
        )
        
        return self.current_com, adjusted_zmp

# Example: Run the walking controller simulation
walking_controller = WalkingController(robot_height=0.8)

print("\nWalking Controller Simulation:")
print("-" * 30)

time_step = 0.02  # 50 Hz control loop
simulation_time = 2.0  # Run for 2 seconds

print(f"Time\tCoM_X\tCoM_Y\tCoM_Z\tZMP_X\tZMP_Y\tSupport")
print("-" * 60)

for t in np.arange(0, simulation_time, time_step):
    com_pos, zmp_pos = walking_controller.update_walking_step(time_step)
    
    if int(t / 0.1) % 5 == 0:  # Print every 0.1 seconds
        print(f"{t:.2f}\t{com_pos[0]:.3f}\t{com_pos[1]:.3f}\t{com_pos[2]:.3f}\t"
              f"{zmp_pos[0]:.3f}\t{zmp_pos[1]:.3f}\t{walking_controller.support_foot}")
```

## Implementation: Simple Walking Pattern

```python
def create_simple_walk_pattern(step_count=10, step_length=0.3, step_width=0.2):
    """
    Create a simple walk pattern with alternating steps.
    
    Args:
        step_count: Number of steps to include in the pattern
        step_length: Forward distance per step
        step_width: Lateral distance between feet
    
    Returns:
        Tuple of (left_foot_positions, right_foot_positions) 
        Each is a list of (x, y) positions over time
    """
    left_positions = []
    right_positions = []
    
    # Starting positions (both feet flat on ground)
    left_x, left_y = 0.0, step_width / 2
    right_x, right_y = 0.0, -step_width / 2
    
    # Add initial positions
    left_positions.append((left_x, left_y))
    right_positions.append((right_x, right_y))
    
    for i in range(step_count):
        if i % 2 == 0:  # Left foot stance, right foot moves forward
            # Move right foot forward
            right_x += step_length
            right_y = -step_width / 2
        else:  # Right foot stance, left foot moves forward
            # Move left foot forward
            left_x += step_length
            left_y = step_width / 2
        
        # Add new positions
        left_positions.append((left_x, left_y))
        right_positions.append((right_x, right_y))
    
    return left_positions, right_positions

def simulate_biped_walking_with_balance_computed():
    """
    Simulate simple biped walking while computing balance metrics.
    """
    # Create walking pattern
    left_pos, right_pos = create_simple_walk_pattern(step_count=8, step_length=0.3, step_width=0.2)
    
    # Calculate CoM trajectory (simplified as midpoint between feet when both on ground)
    com_trajectory = []
    
    for i in range(len(left_pos)):
        left_x, left_y = left_pos[i]
        right_x, right_y = right_pos[i]
        
        # Simplified CoM as midpoint between feet
        com_x = (left_x + right_x) / 2
        com_y = (left_y + right_y) / 2
        
        com_trajectory.append((com_x, com_y))
    
    # Calculate the support polygon (convex hull of feet positions)
    # For simplicity, consider just the x-range of support
    support_polygons = []
    for i in range(len(left_pos)):
        left_x, left_y = left_pos[i]
        right_x, right_y = right_pos[i]
        
        # Define support polygon as the range between feet
        min_x = min(left_x, right_x)
        max_x = max(left_x, right_x)
        min_y = min(left_y, right_y) - 0.1  # Add some width
        max_y = max(left_y, right_y) + 0.1  # Add some width
        
        support_polygons.append(((min_x, max_x), (min_y, max_y)))
    
    # Calculate stability metrics
    stable_positions = 0
    total_positions = len(com_trajectory)
    
    for i in range(total_positions):
        com_x, com_y = com_trajectory[i]
        min_x, max_x = support_polygons[i][0]
        min_y, max_y = support_polygons[i][1]
        
        # Check if CoM is within support polygon
        if min_x <= com_x <= max_x and min_y <= com_y <= max_y:
            stable_positions += 1
    
    stability_ratio = stable_positions / total_positions
    
    return {
        'left_positions': left_pos,
        'right_positions': right_pos,
        'com_trajectory': com_trajectory,
        'stability_ratio': stability_ratio
    }

# Run the simulation
walking_result = simulate_biped_walking_with_balance_computed()

print(f"\nWalking Simulation Results:")
print("-" * 25)
print(f"Total steps: {len(walking_result['left_positions'])}")
print(f"Stability ratio: {walking_result['stability_ratio']:.2%}")

# Show first few positions
print(f"\nFirst 5 positions:")
for i in range(min(5, len(walking_result['left_positions']))):
    left = walking_result['left_positions'][i]
    right = walking_result['right_positions'][i]
    com = walking_result['com_trajectory'][i]
    print(f"  Step {i}: L({left[0]:.2f}, {left[1]:.2f}), "
          f"R({right[0]:.2f}, {right[1]:.2f}), "
          f"CoM({com[0]:.2f}, {com[1]:.2f})")
```

## Summary

This chapter covered the fundamental concepts of gait planning and locomotion for humanoid robots:

1. **Biomechanics**: Understanding the principles of stable walking
2. **ZMP Theory**: The Zero Moment Point as a key control variable
3. **Inverted Pendulum Model**: Simplified dynamics for walking control
4. **Footstep Planning**: How to plan where to place the feet
5. **Walking Controllers**: Feedback control systems to maintain balance during walking

Effective locomotion in humanoids requires carefully orchestrating the movement of multiple limbs while maintaining balance. This involves both motion planning and feedback control to handle disturbances and maintain stable walking.

## Exercises

1. Implement a walking controller that can handle turning motions in addition to straight walking
2. Create a ZMP-based controller that can handle walking on uneven terrain
3. Implement a more sophisticated footstep planner that can avoid obstacles
4. Design a balance controller that uses arm movements to help maintain balance

## References

- Kajita, S. (2005). *Humanoid Robots*. OHM Publisher.
- Vukobratovic, M., & Borrell, A. (1985). *Stability of Humanoid Locomotion*. IEEE Transactions on Systems, Man, and Cybernetics.
- Pratt, J., & Goswami, A. (2007). *On Limit Cycles and Trajectory Tracking by Feedback Linearization of a Compass Gait Biped*. International Journal of Humanoid Robotics.