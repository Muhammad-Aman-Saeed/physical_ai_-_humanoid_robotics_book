---
title: Simulation Basics Tutorial
sidebar_position: 2
---

# Simulation Basics Tutorial

This tutorial introduces the fundamentals of robotics simulation using PyBullet, which is the primary physics simulation environment used throughout this book for Physical AI and Humanoid Robotics examples.

## Introduction to PyBullet

PyBullet is a Python module for physics simulation that provides a rich set of tools for robotics research and development. It offers realistic physics simulation, collision detection, and a wide range of robot models.

### Why Simulation?

Simulation is crucial in robotics for several reasons:
- **Safety**: Test dangerous scenarios without physical risk
- **Cost**: Reduce costs of hardware testing
- **Repeatability**: Same conditions for each experiment
- **Speed**: Run experiments faster than real-time
- **Debugging**: More control over environmental parameters

## Installing and Importing PyBullet

First, make sure PyBullet is installed:

```bash
pip install pybullet
```

Then import it in your Python environment:

```python
import pybullet as p
import pybullet_data
import time
import numpy as np
```

## Basic PyBullet Concepts

PyBullet operates by connecting to a physics server and performing simulation steps. Here are the key concepts:

- **Physics Server**: The engine running the physics simulation
- **URDF**: Universal Robot Description Format, a file format for defining robot models
- **Link**: A single rigid body part of a robot
- **Joint**: A connection between two links
- **Degrees of Freedom (DOF)**: Independent movements of a joint

## Creating Your First Simulation

Let's start with a simple simulation that loads a robot and runs physics:

```python
import pybullet as p
import pybullet_data
import time

# Connect to PyBullet physics server (GUI mode)
physicsClient = p.connect(p.GUI)  # p.DIRECT for non-graphical version

# Set gravity (in m/s^2)
p.setGravity(0, 0, -9.81)

# Load plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# Load a simple robot (KUKA LBR iiwa)
robotStartPos = [0, 0, 0]
robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF("kuka_iiwa/model.urdf", robotStartPos, robotStartOrientation)

# Run simulation for 1000 steps
for i in range(1000):
    # Step the simulation
    p.stepSimulation()
    
    # Small delay to see the simulation in real-time
    time.sleep(1./240.)  # About 240 FPS

# Disconnect from the physics server
p.disconnect()
```

## Understanding Coordinate Systems

PyBullet uses a right-handed coordinate system:
- **X**: Forward/backward
- **Y**: Left/right
- **Z**: Up/down

Rotations are represented using quaternions (x, y, z, w) where w is the scalar component.

## Controlling Joints

Let's create a simulation where we control a robot's joints:

```python
import pybullet as p
import pybullet_data
import time
import numpy as np

# Connect to physics server
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)

# Load plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# Load robot (KUKA LBR iiwa)
robotStartPos = [0, 0, 0]
robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF("kuka_iiwa/model.urdf", robotStartPos, robotStartOrientation)

# Get the number of joints
numJoints = p.getNumJoints(robotId)
print(f"Number of joints: {numJoints}")

# Print joint information
for i in range(numJoints):
    jointInfo = p.getJointInfo(robotId, i)
    print(f"Joint {i}: {jointInfo[1].decode('utf-8')}, Type: {jointInfo[2]}")

# Control the first joint (shoulder)
jointIndex = 1
targetPosition = 0.0  # radians
p.setJointMotorControl2(
    bodyIndex=robotId,
    jointIndex=jointIndex,
    controlMode=p.POSITION_CONTROL,
    targetPosition=targetPosition,
    force=500,  # maximum force/torque
)

# Run simulation with joint control
for i in range(1000):
    # Oscillate the joint
    targetPosition = 0.5 * np.sin(i * 0.01)  # Sinusoidal movement
    p.setJointMotorControl2(
        bodyIndex=robotId,
        jointIndex=jointIndex,
        controlMode=p.POSITION_CONTROL,
        targetPosition=targetPosition,
        force=500,
    )
    
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
```

## Adding Objects to the Simulation

Let's add objects that the robot can interact with:

```python
import pybullet as p
import pybullet_data
import time

# Connect to physics server
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)

# Load plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# Add a box
boxStartPos = [0.5, 0, 0.5]  # Position the box in front of the robot
boxStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("cube.urdf", boxStartPos, boxStartOrientation, 
                   globalScaling=0.5)  # Make box smaller

# Change box color (optional)
p.changeVisualShape(boxId, -1, rgbaColor=[1, 0, 0, 1])  # Red color

# Add a sphere
sphereStartPos = [-0.5, 0.5, 0.5]
sphereId = p.loadURDF("sphere.urdf", sphereStartPos, 
                      globalScaling=0.3)
p.changeVisualShape(sphereId, -1, rgbaColor=[0, 0, 1, 1])  # Blue color

# Run simulation
for i in range(1000):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
```

## Getting Sensor Data

Let's see how to extract information from our simulation:

```python
import pybullet as p
import pybullet_data
import time

# Connect to physics server
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)

# Load plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# Add a free-falling box
boxStartPos = [0, 0, 2]  # Start 2 meters above ground
boxStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("cube.urdf", boxStartPos, boxStartOrientation, 
                   globalScaling=0.2)

print("Initial position:", p.getBasePositionAndOrientation(boxId))

# Run simulation and extract data
for i in range(500):  # Run for 500 steps
    p.stepSimulation()
    
    # Get position and orientation
    position, orientation = p.getBasePositionAndOrientation(boxId)
    
    # Print position every 50 steps
    if i % 50 == 0:
        print(f"Step {i}: Position = {position}")
    
    # Check if the box has hit the ground
    if position[2] < 0.12:  # Cube has 0.2 size, so it's almost touching ground
        print(f"Box hit ground at step {i} with position: {position}")
        break
    
    time.sleep(1./240.)

p.disconnect()
```

## Collision Detection

Here's an example of how to detect collisions between objects:

```python
import pybullet as p
import pybullet_data
import time

# Connect to physics server
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)

# Load plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# Add two boxes that will collide
box1StartPos = [0, 0, 1]
box1Id = p.loadURDF("cube.urdf", box1StartPos, 
                    globalScaling=0.2, useMaximalCoordinates=0)

box2StartPos = [0, 0, 3]  # Start higher
box2Id = p.loadURDF("cube.urdf", box2StartPos, 
                    globalScaling=0.2, useMaximalCoordinates=0)

print("Starting simulation...")

# Run simulation and check for collisions
for i in range(1000):
    p.stepSimulation()
    
    # Check for collisions between the two boxes
    contact_points = p.getContactPoints(bodyA=box1Id, bodyB=box2Id)
    
    if len(contact_points) > 0:
        print(f"Collision detected at step {i}!")
        print(f"Contact points: {len(contact_points)}")
        break
    
    time.sleep(1./240.)

p.disconnect()
```

## Advanced: Creating Custom Objects

You can create objects directly in PyBullet without loading a URDF:

```python
import pybullet as p
import time

# Connect to physics server
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)

# Create a ground plane
planeShape = p.createCollisionShape(p.GEOM_PLANE)
planeId = p.createMultiBody(0, planeShape)  # mass=0 means static

# Create a box
boxHalfSize = [0.2, 0.2, 0.2]
boxShape = p.createCollisionShape(p.GEOM_BOX, halfExtents=boxHalfSize)
boxVisualShape = p.createVisualShape(p.GEOM_BOX, 
                                    halfExtents=boxHalfSize,
                                    rgbaColor=[1, 0, 0, 1])  # Red

# Create a multi-body with both collision and visual shapes
boxId = p.createMultiBody(
    baseMass=1,  # 1 kg
    baseCollisionShapeIndex=boxShape,
    baseVisualShapeIndex=boxVisualShape,
    basePosition=[0, 0, 2]  # Start 2 meters high
)

# Run simulation
for i in range(1000):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
```

## Working with Inverse Kinematics

For more advanced robotics applications, you often need inverse kinematics:

```python
import pybullet as p
import pybullet_data
import time
import numpy as np

# Connect to physics server
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)

# Load plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# Load a robot arm
robotStartPos = [0, 0, 0]
robotId = p.loadURDF("kuka_iiwa/model.urdf", robotStartPos, 
                     p.getQuaternionFromEuler([0, 0, 0]))

# Define the end effector link index (for KUKA iiwa, it's typically the last link)
endEffectorId = 6  # Check getNumJoints() to confirm

# Target position for the end effector
targetPos = [0.5, 0, 0.5]

# Run inverse kinematics to find joint positions
jointPoses = p.calculateInverseKinematics(
    robotId,
    endEffectorId,
    targetPos,
    maxNumIterations=100,
    residualThreshold=0.001
)

# Apply the calculated joint positions
for i in range(len(jointPoses)):
    p.setJointMotorControl2(
        bodyIndex=robotId,
        jointIndex=i,
        controlMode=p.POSITION_CONTROL,
        targetPosition=jointPoses[i],
        force=500
    )

# Run simulation to see the result
for i in range(500):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
```

## Best Practices for Simulation

1. **Start Simple**: Begin with basic objects and gradually add complexity
2. **Check Gravity**: Make sure gravity is set appropriately for your scenario
3. **Time Steps**: Use appropriate time steps (1/240 is often a good default)
4. **Debug Mode**: Use GUI mode during development, switch to DIRECT for performance
5. **Reset Simulation**: Use `p.resetSimulation()` when needed to clean up

## Common Pitfalls to Avoid

1. **Forgetting to Step Simulation**: Always call `p.stepSimulation()` in your loop
2. **Not Disconnecting**: Always call `p.disconnect()` to free resources
3. **Incorrect Units**: PyBullet uses SI units (meters, seconds, etc.)
4. **Unstable Simulations**: Adjust physics parameters if objects behave unrealistically

## Summary

This tutorial covered the basics of PyBullet simulation:
- Connecting to and disconnecting from the physics server
- Loading and positioning objects
- Controlling joints and applying forces
- Extracting sensor data and detecting collisions
- Using inverse kinematics for robot control

With this foundation, you can now work with more complex simulations in the Physical AI and Humanoid Robotics examples throughout this book.