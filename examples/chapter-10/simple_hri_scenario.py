"""
Simple HRI scenario example
File: examples/chapter-10/simple_hri_scenario.py

This example demonstrates a simple human-robot interaction scenario
where a humanoid robot assists a person in a home environment. The
scenario includes basic dialogue management, gesture recognition,
and appropriate robot responses based on social robotics principles.
"""

import time
import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np


class InteractionState(Enum):
    IDLE = 1
    LISTENING = 2
    PROCESSING = 3
    RESPONDING = 4
    GESTURING = 5
    FOLLOWING = 6


class RobotRole(Enum):
    ASSISTANT = 1
    COMPANION = 2
    INSTRUCTOR = 3


@dataclass
class UserCommand:
    """Represents a command from the user"""
    text: str
    intent: str  # The identified intent of the user
    confidence: float  # Confidence level of intent recognition
    entities: Dict[str, str]  # Identified entities (e.g., time, location, object)


@dataclass
class RobotAction:
    """Represents an action the robot should perform"""
    action_type: str  # 'speak', 'gesture', 'move', etc.
    content: str  # The content of the action
    priority: int  # Priority level (1-5)
    delay: float  # Delay before execution in seconds


class SimpleDialogueManager:
    """Simple dialogue manager for the robot"""
    
    def __init__(self):
        # Define simple intent patterns
        self.intent_patterns = {
            'greeting': [
                'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'
            ],
            'farewell': [
                'goodbye', 'bye', 'see you', 'farewell', 'good night'
            ],
            'request_assistance': [
                'can you help', 'help me', 'assist', 'i need help', 'can you', 'could you'
            ],
            'ask_time': [
                'what time', 'current time', 'time is it', 'tell me the time'
            ],
            'ask_date': [
                'what date', 'current date', 'what is the date', 'date today'
            ],
            'ask_weather': [
                'weather', 'how is the weather', 'is it raining', 'temperature'
            ],
            'move_to_location': [
                'go to', 'move to', 'walk to', 'navigate to', 'find', 'go find'
            ],
            'fetch_object': [
                'get', 'bring', 'fetch', 'bring me', 'get me', 'pick up'
            ],
            'follow_me': [
                'follow me', 'come with me', 'follow', 'accompany me', 'walk with me'
            ],
            'tell_joke': [
                'tell me a joke', 'make me laugh', 'joke', 'funny'
            ]
        }
        
        # Predefined responses for different intents
        self.responses = {
            'greeting': [
                "Hello! How can I assist you today?",
                "Hi there! What can I do for you?",
                "Good to see you! How are you doing?"
            ],
            'farewell': [
                "Goodbye! Have a great day!",
                "See you later!",
                "Take care!"
            ],
            'request_assistance': [
                "I'm here to help. What do you need assistance with?",
                "How can I assist you?",
                "What can I help you with?"
            ],
            'ask_time': [
                f"The current time is {time.strftime('%H:%M')}.",
                f"It's {time.strftime('%I:%M %p')} currently."
            ],
            'ask_date': [
                f"Today is {time.strftime('%A, %B %d, %Y')}.",
                f"The date is {time.strftime('%m/%d/%Y')}."
            ],
            'ask_weather': [
                "I don't have real-time weather data, but I can help you check a weather app.",
                "I'm unable to access current weather information, sorry."
            ],
            'move_to_location': [
                "I'll navigate to that location for you.",
                "On my way to that location.",
                "I can go there for you. Please wait."
            ],
            'fetch_object': [
                "I'll fetch that for you.",
                "I'll go and bring that to you.",
                "I can retrieve that object for you."
            ],
            'follow_me': [
                "I'll follow you now.",
                "Right behind you!",
                "Following you now."
            ],
            'tell_joke': [
                "Why don't scientists trust atoms? Because they make up everything!",
                "I'm not very good at jokes, but here's one: What do you call a fake noodle? An impasta!",
                "Why did the scarecrow win an award? Because he was outstanding in his field!"
            ]
        }
    
    def identify_intent(self, text: str) -> tuple[str, float]:
        """Identify the intent of the user's input"""
        text_lower = text.lower()
        best_intent = 'unknown'
        best_confidence = 0.0
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    # Simple confidence based on pattern length and matches
                    confidence = min(1.0, len(pattern) / 10.0 + 0.2)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = intent
        
        return best_intent, best_confidence
    
    def generate_response(self, intent: str) -> str:
        """Generate an appropriate response for the given intent"""
        if intent in self.responses:
            return random.choice(self.responses[intent])
        else:
            return "I'm not sure how to respond to that. Can you rephrase?"


class GestureController:
    """Simple gesture controller for the robot"""
    
    def __init__(self):
        self.available_gestures = [
            'wave', 'nod', 'shake_head', 'point', 'clap', 'peace_sign', 'thumbs_up', 'beckon'
        ]
        self.current_gesture = None
    
    def recognize_gesture(self, gesture_input: str) -> Optional[str]:
        """Recognize a gesture from input (simulated)"""
        # In a real system, this would process sensor data
        gesture_input_lower = gesture_input.lower()
        
        if 'wave' in gesture_input_lower:
            return 'wave'
        elif 'nod' in gesture_input_lower:
            return 'nod'
        elif 'shake' in gesture_input_lower or 'head' in gesture_input_lower:
            return 'shake_head'
        elif 'point' in gesture_input_lower:
            return 'point'
        elif 'clap' in gesture_input_lower:
            return 'clap'
        elif 'peace' in gesture_input_lower:
            return 'peace_sign'
        elif 'thumb' in gesture_input_lower or 'thumbs' in gesture_input_lower:
            return 'thumbs_up'
        elif 'beckon' in gesture_input_lower or 'come' in gesture_input_lower:
            return 'beckon'
        
        return None
    
    def execute_gesture(self, gesture_name: str) -> bool:
        """Execute a gesture (simulated)"""
        if gesture_name in self.available_gestures:
            print(f"*Robot performs '{gesture_name}' gesture*")
            self.current_gesture = gesture_name
            return True
        else:
            print(f"Robot doesn't know how to perform '{gesture_name}' gesture")
            return False


class HRIEngine:
    """Main engine for Human-Robot Interaction"""
    
    def __init__(self, robot_name: str = "ARIA", role: RobotRole = RobotRole.ASSISTANT):
        self.robot_name = robot_name
        self.role = role
        self.state = InteractionState.IDLE
        self.dialogue_manager = SimpleDialogueManager()
        self.gesture_controller = GestureController()
        self.user_context = {}  # Store information about the user
        self.conversation_history = []
        
    def process_user_input(self, user_input: str) -> List[RobotAction]:
        """Process user input and generate robot actions"""
        # Identify intent
        intent, confidence = self.dialogue_manager.identify_intent(user_input)
        
        # Store in conversation history
        command = UserCommand(text=user_input, intent=intent, confidence=confidence, entities={})
        self.conversation_history.append(command)
        
        # Generate appropriate actions based on intent
        actions = []
        
        if intent == 'greeting' and confidence > 0.3:
            response = self.dialogue_manager.generate_response(intent)
            actions.append(RobotAction('speak', response, 3, 0.0))
            actions.append(RobotAction('gesture', 'wave', 2, 0.5))
        
        elif intent == 'request_assistance' and confidence > 0.4:
            response = self.dialogue_manager.generate_response(intent)
            actions.append(RobotAction('speak', response, 3, 0.0))
        
        elif intent in ['ask_time', 'ask_date'] and confidence > 0.4:
            response = self.dialogue_manager.generate_response(intent)
            actions.append(RobotAction('speak', response, 3, 0.0))
        
        elif intent == 'follow_me' and confidence > 0.5:
            response = self.dialogue_manager.generate_response(intent)
            actions.append(RobotAction('speak', response, 3, 0.0))
            actions.append(RobotAction('follow', 'user', 5, 1.0))
            self.state = InteractionState.FOLLOWING
        
        elif intent == 'tell_joke' and confidence > 0.5:
            response = self.dialogue_manager.generate_response(intent)
            actions.append(RobotAction('speak', response, 2, 0.0))
            actions.append(RobotAction('gesture', 'thumbs_up', 2, 1.0))
        
        else:
            # Default response
            actions.append(RobotAction('speak', "I'm not sure I understand. Could you repeat that?", 3, 0.0))
        
        return actions
    
    def execute_actions(self, actions: List[RobotAction]):
        """Execute the robot actions"""
        for action in actions:
            print(f"\n[Action: {action.action_type}] {action.content}")
            
            if action.action_type == 'speak':
                print(f"{self.robot_name}: {action.content}")
                time.sleep(action.delay)
                
            elif action.action_type == 'gesture':
                self.gesture_controller.execute_gesture(action.content)
                time.sleep(action.delay)
                
            elif action.action_type == 'follow':
                print(f"*Robot begins following {action.content}*")
                time.sleep(action.delay)
            
            # Respect action priority and timing
            time.sleep(0.5)
    
    def start_interaction(self):
        """Start the HRI scenario"""
        print(f"Starting HRI scenario with {self.robot_name} ({self.role.name})")
        print(f"{self.robot_name}: Hello! I'm {self.robot_name}, your {self.role.value} robot. How can I help you today?")
        
        # Simulate a basic interaction scenario
        scenarios = [
            "Hello ARIA!",
            "What time is it?",
            "Can you help me?",
            "Follow me to the kitchen",
            "Can you tell me a joke?",
        ]
        
        for i, scenario in enumerate(scenarios):
            print(f"\n--- Scenario {i+1}: '{scenario}' ---")
            
            # Process user input
            actions = self.process_user_input(scenario)
            
            # Execute actions
            self.execute_actions(actions)
            
            # Small delay between scenarios
            time.sleep(1)
        
        print(f"\n{self.robot_name}: Thank you for interacting with me today!")
        print("HRI scenario completed.")
        
        # Print conversation summary
        print(f"\n--- Conversation Summary ---")
        print(f"Total interactions: {len(self.conversation_history)}")
        intents_used = [cmd.intent for cmd in self.conversation_history]
        unique_intents = set(intents_used)
        print(f"Intents recognized: {', '.join(unique_intents)}")


def simulate_user_behavior(robot: HRIEngine, user_commands: List[str]):
    """Simulate user behavior for the HRI scenario"""
    print("\n--- Starting simulated user interactions ---")
    
    for i, command in enumerate(user_commands):
        print(f"\nUser: {command}")
        
        # Process the user command
        actions = robot.process_user_input(command)
        
        # Execute robot actions
        robot.execute_actions(actions)
        
        # Random delay to simulate real interaction
        time.sleep(random.uniform(0.5, 1.5))


def main():
    """
    Main function to run the HRI scenario example
    """
    print("Starting Simple HRI Scenario Example")
    print("=" * 50)
    
    # Create HRI engine
    hri_robot = HRIEngine(robot_name="ARIA", role=RobotRole.ASSISTANT)
    
    # Run the demonstration
    hri_robot.start_interaction()
    
    # Additional simulated interactions
    print("\n" + "=" * 50)
    print("Additional simulated interactions:")
    
    # Define a set of simulated user commands
    user_commands = [
        "Hi there, ARIA!",
        "What date is it today?",
        "Could you follow me?",
        "ARIA, can you help me find my keys?",
        "Tell me a joke to cheer me up"
    ]
    
    # Run simulated user interactions
    simulate_user_behavior(hri_robot, user_commands)
    
    print("\nSimple HRI Scenario Example completed!")


if __name__ == "__main__":
    main()