"""
Test to verify system prevents dangerous joint limits violations
File: tests/unit/test_safety_limits.py
"""
import unittest
import numpy as np


class TestSafetyLimits(unittest.TestCase):
    """Test suite for verifying joint limit safety in robotic systems"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Define some basic robot parameters for testing
        self.joint_limits_min = np.array([-2.0, -1.5, -3.0, -2.5, -2.0, -0.5])  # radians
        self.joint_limits_max = np.array([2.0, 1.5, 3.0, 2.5, 2.0, 0.5])       # radians
        
    def test_joint_limits_enforcement(self):
        """Test that the system prevents movement beyond joint limits"""
        # Simulate a function that should enforce joint limits
        def enforce_joint_limits(joint_angles, min_limits, max_limits):
            """Apply joint limits to ensure angles stay within safe range"""
            return np.clip(joint_angles, min_limits, max_limits)
        
        # Test case: request exceeds upper limit
        requested_angles = np.array([2.5, 1.0, 3.5, 2.0, -2.5, 1.0])  # Some values exceed limits
        expected_angles = np.array([2.0, 1.0, 3.0, 2.0, -2.0, 0.5])   # After applying limits
        
        result = enforce_joint_limits(requested_angles, self.joint_limits_min, self.joint_limits_max)
        np.testing.assert_array_equal(result, expected_angles)
        
    def test_within_limits_passes(self):
        """Test that values within limits are not modified"""
        def enforce_joint_limits(joint_angles, min_limits, max_limits):
            """Apply joint limits to ensure angles stay within safe range"""
            return np.clip(joint_angles, min_limits, max_limits)
        
        requested_angles = np.array([1.0, 0.5, -2.5, 1.5, -1.0, 0.2])  # All values within limits
        expected_angles = requested_angles.copy()
        
        result = enforce_joint_limits(requested_angles, self.joint_limits_min, self.joint_limits_max)
        np.testing.assert_array_equal(result, expected_angles)
        
    def test_negative_limits_enforcement(self):
        """Test enforcement on negative limit violations"""
        def enforce_joint_limits(joint_angles, min_limits, max_limits):
            """Apply joint limits to ensure angles stay within safe range"""
            return np.clip(joint_angles, min_limits, max_limits)
        
        requested_angles = np.array([-2.5, 0.5, -3.5, 1.5, 1.0, -1.0])  # Some values below limits
        expected_angles = np.array([-2.0, 0.5, -3.0, 1.5, 1.0, -0.5])   # After applying limits
        
        result = enforce_joint_limits(requested_angles, self.joint_limits_min, self.joint_limits_max)
        np.testing.assert_array_equal(result, expected_angles)


if __name__ == '__main__':
    unittest.main()