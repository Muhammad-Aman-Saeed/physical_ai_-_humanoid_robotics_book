---
title: Setup Environment Tutorial
sidebar_position: 1
---

# Setup Environment Tutorial

This tutorial will guide you through setting up your development environment to work with the concepts and code examples in this book on Physical AI and Humanoid Robotics.

## Prerequisites

Before beginning the setup, ensure you have the following:

- **Operating System**: Linux (Ubuntu 22.04 LTS recommended) or macOS 10.15+
- **Windows users**: WSL2 with Ubuntu 22.04 is recommended, or use Docker for containerized environments
- **RAM**: 8GB minimum, 16GB or more recommended
- **CPU**: Multi-core processor with good performance for simulation
- **GPU**: Optional but recommended for advanced simulations (CUDA-capable if using Isaac Gym)

## Python Environment Setup

### 1. Install Python

Ensure you have Python 3.9 or higher installed:

```bash
python3 --version
```

If Python is not installed or you have an older version:

**On Ubuntu:**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

**On macOS:**
```bash
# Using Homebrew
brew install python3
```

### 2. Create a Virtual Environment

```bash
# Create a virtual environment
python3 -m venv physical_ai_env

# Activate the environment
source physical_ai_env/bin/activate  # On Windows: physical_ai_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Required Python Packages

Create a requirements file with the packages used throughout the book:

```bash
# Create requirements.txt
cat > requirements.txt << EOF
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
jupyter>=1.0.0
pybullet>=3.2.5
roslibpy>=1.3.0
opencv-python>=4.5.0
mediapipe>=0.8.0
tensorflow>=2.8.0
torch>=1.10.0
pandas>=1.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
speechrecognition>=3.8.1
pyttsx3>=2.90
nltk>=3.7
EOF

# Install the packages
pip install -r requirements.txt
```

## ROS 2 Installation (Optional but Recommended)

ROS 2 (Robot Operating System 2) is essential for many robotics applications. While not required for all examples in this book, installing it will allow you to work with more advanced examples.

### 1. Install ROS 2 Humble Hawksbill (Ubuntu 22.04)

```bash
# Set locale
locale  # check for UTF-8
sudo locale-gen en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS 2 apt repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install ros-humble-cv-bridge ros-humble-tf2-tools ros-humble-tf2-geometry-msgs
```

### 2. Source ROS 2

```bash
# Add to your shell's startup script
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Simulation Environments

### PyBullet Installation

PyBullet is our recommended physics simulation environment:

```bash
pip install pybullet
```

### Testing PyBullet Installation

```python
import pybullet as p
import pybullet_data

# Connect to physics server
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version

# Load plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# Set gravity
p.setGravity(0, 0, -10)

# Add a cube
cubeStartPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("cube.urdf", cubeStartPos, cubeStartOrientation)

# Run simulation for a few steps
for i in range(1000):
    p.stepSimulation()
    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    
p.disconnect()
```

### Mujoco Installation (Optional)

Mujoco is a more advanced physics simulator:

```bash
# Install from PyPI
pip install mujoco

# Or for the latest version
pip install "mujoco>=2.3.0"
```

## Development Tools

### 1. Install Jupyter Notebook/Lab

```bash
pip install jupyter jupyterlab
```

### 2. Install Code Editor

We recommend VS Code with the following extensions:

```bash
# Install VS Code
sudo snap install code --classic

# Recommended extensions
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-iot.vscode-ros
code --install-extension ms-vscode.cpptools
```

### 3. Git Setup

```bash
sudo apt install git  # On Ubuntu
git --version
```

Configure your Git identity:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Setting up the Project Directory

### 1. Create Project Structure

```bash
mkdir physical_ai_project
cd physical_ai_project

# Create directory structure
mkdir -p {docs,examples,exercises,notebooks,src,tests,static}
mkdir -p examples/chapter_{1..12}
mkdir -p exercises/chapter_{1..12}
mkdir -p notebooks/chapter_{1..12}
mkdir -p tests/{unit,integration}
```

### 2. Initialize Git Repository

```bash
git init
```

### 3. Create Basic Project Files

```bash
# Create .gitignore
cat > .gitignore << EOF
# Virtual environments
venv/
env/
ENV/
.venv/
env.bak/
.ENV/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# ROS
devel/
install/
build/
logs/

# OS
.DS_Store
Thumbs.db
EOF

# Create README.md
cat > README.md << EOF
# Physical AI & Humanoid Robotics Project

This project contains code and examples from the book "Physical AI & Humanoid Robotics".
EOF
```

## Testing Your Setup

Let's run a simple test to verify your environment is working correctly:

```python
# test_setup.py
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV
import sys

print("Python version:", sys.version)
print("NumPy version:", np.__version__)

# Test basic NumPy functionality
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = a + b
print("NumPy test - a + b =", c)

# Test matplotlib
plt.figure(figsize=(5, 3))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title("Test Plot")
plt.savefig("test_plot.png")
print("Matplotlib test - saved test_plot.png")

# Test OpenCV
img = np.zeros((100, 100, 3), dtype=np.uint8)
img = cv2.rectangle(img, (10, 10), (90, 90), (255, 0, 0), 2)
cv2.imwrite("test_cv_image.png", img)
print("OpenCV test - saved test_cv_image.png")

print("\nEnvironment setup verification complete!")
print("Your environment is ready to work with the examples in the Physical AI & Humanoid Robotics book.")
```

Run the test:

```bash
python test_setup.py
```

## Troubleshooting Common Issues

### 1. PyBullet Installation Issues

If you encounter issues with PyBullet:

```bash
# For specific architectures
pip install --upgrade --force-reinstall pybullet

# If you're on Apple Silicon Mac, you might need:
pip install --prefer-binary pybullet
```

### 2. ROS 2 Installation Issues

If ROS 2 packages fail to install:

```bash
# Make sure your system is fully updated
sudo apt update && sudo apt upgrade

# Check your locale settings
locale
```

### 3. CUDA Issues

If using GPU acceleration:

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Install appropriate PyTorch for CUDA
# Visit pytorch.org for the correct command for your CUDA version
```

## Next Steps

After completing this setup, you are ready to:

1. Work through the interactive Jupyter notebooks in the `notebooks/` directory
2. Run the code examples in the `examples/` directory
3. Complete the exercises in the `exercises/` directory
4. Explore the concepts in the documentation

## Keeping Your Environment Updated

Regularly update your packages:

```bash
pip list --outdated  # Check for outdated packages
pip install --upgrade -r requirements.txt  # Update to latest versions in requirements
```

For ROS 2 updates:

```bash
sudo apt update
sudo apt upgrade
```

This completes the environment setup for working with Physical AI and Humanoid Robotics concepts. Your development environment is now ready to run all the examples and experiments in this book.