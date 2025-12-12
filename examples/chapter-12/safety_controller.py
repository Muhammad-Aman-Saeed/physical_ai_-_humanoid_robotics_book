"""
Safety controller implementation
File: examples/chapter-12/safety_controller.py

This example implements a safety controller for humanoid robots that monitors
critical parameters and ensures the robot remains in safe operational states.
The controller implements multiple safety layers including joint limits,
velocity limits, and collision avoidance.
"""

import numpy as np
import time
import threading
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


class SafetyState(Enum):
    NORMAL = 1
    WARNING = 2
    EMERGENCY_STOP = 3
    SAFE_STATE = 4


@dataclass
class RobotState:
    """Represents the current state of the robot"""
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    joint_names: List[str]
    robot_position: np.ndarray  # x, y, z position in world
    robot_orientation: np.ndarray  # quaternion [x, y, z, w]
    imu_data: np.ndarray  # [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    proximity_sensors: np.ndarray  # distances to obstacles
    external_forces: np.ndarray  # forces applied to robot
    joint_temperatures: np.ndarray  # temperatures for each joint


class SafetyController:
    """
    A comprehensive safety controller for humanoid robots implementing
    multiple safety layers and protocols.
    """
    
    def __init__(self):
        # Safety configuration
        self.safety_state = SafetyState.NORMAL
        self.emergency_stop_triggered = False
        
        # Joint safety limits
        self.joint_position_limits = {
            'hip_pitch': (-1.5, 1.5),
            'knee_pitch': (0.0, 2.5),
            'ankle_pitch': (-0.5, 0.5),
            'hip_roll': (-0.5, 0.5),
            'ankle_roll': (-0.5, 0.5),
            'shoulder_pitch': (-2.0, 2.0),
            'shoulder_roll': (-1.5, 1.0),
            'elbow_pitch': (-2.0, 0.5),
        }
        
        self.velocity_limit = 2.0  # rad/s
        self.torque_limits = {
            'hip': 100.0,    # Nm
            'knee': 80.0,
            'ankle': 40.0,
            'shoulder': 60.0,
            'elbow': 30.0,
        }
        
        self.proximity_threshold = 0.5  # meters
        self.temperature_threshold = 75.0  # Celsius
        self.fall_threshold = 0.3  # Radians from upright
        
        # Safe position to move to in emergency
        self.safe_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Simplified
        
        # Logging
        self.safety_log = []
        
    def check_joint_limits(self, state: RobotState) -> Tuple[bool, str]:
        """Check if all joint positions are within safe limits"""
        for i, name in enumerate(state.joint_names):
            if name in self.joint_position_limits:
                pos = state.joint_positions[i]
                min_limit, max_limit = self.joint_position_limits[name]
                
                if pos < min_limit or pos > max_limit:
                    msg = f"Joint {name} position limit violated: {pos:.3f} (limits: {min_limit:.3f}, {max_limit:.3f})"
                    return False, msg
        
        return True, "All joint positions within limits"
    
    def check_velocity_limits(self, state: RobotState) -> Tuple[bool, str]:
        """Check if all joint velocities are within safe limits"""
        max_velocity = np.max(np.abs(state.joint_velocities))
        
        if max_velocity > self.velocity_limit:
            return False, f"Velocity limit exceeded: {max_velocity:.3f} rad/s (limit: {self.velocity_limit})"
        
        return True, "All velocities within limits"
    
    def check_torque_limits(self, state: RobotState) -> Tuple[bool, str]:
        """Check if all joint torques are within safe limits"""
        for i, name in enumerate(state.joint_names):
            # Determine torque limit based on joint type
            torque = abs(state.joint_torques[i])
            limit = self.torque_limits.get(name.split('_')[0], 50.0)  # Default to 50 Nm
            
            if torque > limit:
                return False, f"Torque limit exceeded for {name}: {torque:.3f} Nm (limit: {limit})"
        
        return True, "All torques within limits"
    
    def check_proximity_safety(self, state: RobotState) -> Tuple[bool, str]:
        """Check if the robot is at a safe distance from obstacles"""
        if len(state.proximity_sensors) == 0:
            return True, "No proximity sensors available"
        
        min_distance = np.min(state.proximity_sensors)
        
        if min_distance < self.proximity_threshold:
            return False, f"Robot too close to obstacle: {min_distance:.3f}m (threshold: {self.proximity_threshold}m)"
        
        return True, "Safe distance to obstacles"
    
    def check_temperature_safety(self, state: RobotState) -> Tuple[bool, str]:
        """Check if all joint temperatures are within safe limits"""
        max_temp = np.max(state.joint_temperatures)
        
        if max_temp > self.temperature_threshold:
            return False, f"Temperature limit exceeded: {max_temp:.1f}°C (threshold: {self.temperature_threshold}°C)"
        
        return True, "All temperatures within limits"
    
    def check_balance(self, state: RobotState) -> Tuple[bool, str]:
        """Check if the robot is maintaining balance (simple implementation)"""
        # Calculate tilt from upright position using IMU data
        # For simplicity, we'll check the z-component of the accelerometer
        # In a real implementation, we'd use more sophisticated balance metrics
        accel_z = state.imu_data[2]  # z component of acceleration
        
        # If accelerometer z is too far from 9.81 (gravity), the robot may be falling
        if abs(abs(accel_z) - 9.81) > 5.0:  # Significant deviation from gravity
            return False, f"Balance potentially compromised: accel_z = {accel_z:.2f}"
        
        # Check orientation (simplified - real check would be more complex)
        # Looking at the quaternion to determine if robot is too tilted
        orientation = state.imu_data[3:]  # Assuming orientation data is in the second half
        if len(orientation) >= 4:
            # Calculate angle from upright using quaternion
            # For a unit quaternion [x, y, z, w], the rotation angle is 2*arccos(w)
            w = orientation[3]  # w component of quaternion
            angle_from_upright = 2 * math.acos(min(max(w, -1.0), 1.0))  # Clamp to valid range
            
            if angle_from_upright > self.fall_threshold:
                return False, f"Robot tilt angle too large: {math.degrees(angle_from_upright):.1f}° (threshold: {math.degrees(self.fall_threshold):.1f}°)"
        
        return True, "Balance appears stable"
    
    def run_safety_checks(self, robot_state: RobotState) -> Tuple[SafetyState, List[str]]:
        """
        Run all safety checks on the provided robot state
        Returns the safety state and a list of safety messages
        """
        # Perform all safety checks
        checks = [
            self.check_joint_limits(robot_state),
            self.check_velocity_limits(robot_state),
            self.check_torque_limits(robot_state),
            self.check_proximity_safety(robot_state),
            self.check_temperature_safety(robot_state),
            self.check_balance(robot_state)
        ]
        
        # Evaluate results
        all_safe = True
        safety_messages = []
        
        for is_safe, message in checks:
            if not is_safe:
                all_safe = False
                safety_messages.append(message)
        
        # Update safety state based on results
        if not all_safe:
            if self.safety_state == SafetyState.NORMAL:
                self.safety_state = SafetyState.WARNING
            elif self.safety_state == SafetyState.WARNING:
                self.safety_state = SafetyState.EMERGENCY_STOP
            elif self.safety_state in [SafetyState.EMERGENCY_STOP, SafetyState.SAFE_STATE]:
                self.safety_state = SafetyState.SAFE_STATE
        else:
            # If all checks pass, reset to normal state (if not in safe state from previous emergency)
            if self.safety_state != SafetyState.SAFE_STATE:
                self.safety_state = SafetyState.NORMAL
            safety_messages = ["All safety checks passed"]
        
        # Log safety state changes
        self.safety_log.append({
            'timestamp': time.time(),
            'state': self.safety_state,
            'messages': safety_messages.copy()
        })
        
        return self.safety_state, safety_messages
    
    def transition_to_safe_state(self, robot_interface=None) -> bool:
        """
        Transition the robot to a safe state
        In a real implementation, this would interface with the robot's control system
        """
        print(f"Transitioning to safe state from {self.safety_state}")
        
        if self.safety_state in [SafetyState.EMERGENCY_STOP, SafetyState.SAFE_STATE]:
            # Move to predefined safe position if robot interface provided
            if robot_interface:
                robot_interface.move_to_position(self.safe_position)
                robot_interface.disable_actuators()
            
            print("Robot is in safe state")
            return True
        else:
            print("No safe state transition needed")
            return False
    
    def get_safety_status(self) -> dict:
        """Get current safety status information"""
        return {
            'current_state': self.safety_state,
            'emergency_stop': self.emergency_stop_triggered,
            'last_log_entries': self.safety_log[-5:] if self.safety_log else []  # Last 5 logs
        }


class SimulationRobotInterface:
    """
    A simulated robot interface to demonstrate the safety controller
    """
    
    def __init__(self):
        # Initialize robot state
        self.joint_names = ['hip_pitch', 'knee_pitch', 'ankle_pitch', 'hip_roll', 'ankle_roll']
        self.joint_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.joint_velocities = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.joint_torques = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.joint_temperatures = np.array([25.0, 25.0, 25.0, 25.0, 25.0])  # Celsius
        
        # Position and orientation
        self.position = np.array([0.0, 0.0, 0.8])  # Standing at 0.8m height
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Unit quaternion (no rotation)
        
        # IMU data: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        self.imu_data = np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])  # Gravity reading when upright
        
        # Proximity sensors (distances to obstacles in meters)
        self.proximity_sensors = np.array([1.0, 1.0, 1.0, 1.0])  # 4 sensors in different directions
        
        # External forces (in a real robot, these would come from force/torque sensors)
        self.external_forces = np.array([0.0, 0.0, 0.0])
    
    def move_to_position(self, position):
        """Simulate moving robot to a new position"""
        print(f"Moving robot to safe position: {position}")
        # In a real implementation, this would command the robot's actuators
    
    def disable_actuators(self):
        """Simulate disabling robot actuators for safety"""
        print("Disabling all robot actuators for safety")
        # In a real implementation, this would cut power to the actuators
    
    def generate_random_state(self):
        """Generate a random robot state for demonstration"""
        # Slightly randomize joint positions
        self.joint_positions = np.array([
            np.random.uniform(-0.5, 0.5) for _ in range(len(self.joint_names))
        ])
        
        # Randomize velocities (keeping within reasonable bounds)
        self.joint_velocities = np.array([
            np.random.uniform(-1.0, 1.0) for _ in range(len(self.joint_names))
        ])
        
        # Randomize torques (keeping within reasonable bounds)
        self.joint_torques = np.array([
            np.random.uniform(-20.0, 20.0) for _ in range(len(self.joint_names))
        ])
        
        # Randomize temperatures (keeping within safe bounds, but occasionally unsafe)
        self.joint_temperatures = np.array([
            np.random.uniform(25.0, 60.0) for _ in range(len(self.joint_names))
        ])
        
        # Randomize proximity sensors (occasionally have close objects)
        self.proximity_sensors = np.array([
            np.random.uniform(0.3, 2.0) for _ in range(4)
        ])
        
        return RobotState(
            joint_positions=self.joint_positions,
            joint_velocities=self.joint_velocities,
            joint_torques=self.joint_torques,
            joint_names=self.joint_names,
            robot_position=self.position,
            robot_orientation=self.orientation,
            imu_data=self.imu_data,
            proximity_sensors=self.proximity_sensors,
            external_forces=self.external_forces,
            joint_temperatures=self.joint_temperatures
        )


def main():
    """
    Main function to demonstrate the safety controller
    """
    print("Initializing safety controller for humanoid robot...")
    
    # Create safety controller and robot interface
    safety_controller = SafetyController()
    robot_interface = SimulationRobotInterface()
    
    print("\nRunning safety checks in simulated environment...")
    
    # Run simulation for a few iterations
    for i in range(10):
        print(f"\n--- Iteration {i+1} ---")
        
        # Generate random robot state
        robot_state = robot_interface.generate_random_state()
        
        # Run safety checks
        safety_state, messages = safety_controller.run_safety_checks(robot_state)
        
        print(f"Current safety state: {safety_state.name}")
        for msg in messages:
            print(f"  - {msg}")
        
        # If emergency state is triggered, transition to safe state
        if safety_state in [SafetyState.EMERGENCY_STOP, SafetyState.SAFE_STATE]:
            print("Emergency detected! Transitioning to safe state...")
            safety_controller.transition_to_safe_state(robot_interface)
            break
        
        # Small delay to simulate real-time operation
        time.sleep(0.1)
    
    # Print final safety status
    print("\nFinal safety status:")
    status = safety_controller.get_safety_status()
    print(f"  Current state: {status['current_state'].name}")
    print(f"  Emergency stop triggered: {status['emergency_stop']}")
    
    print("\nSafety controller demonstration completed!")


if __name__ == "__main__":
    main()