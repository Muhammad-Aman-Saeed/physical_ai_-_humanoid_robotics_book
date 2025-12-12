"""
Basic conversational agent exercise
File: exercises/chapter-10/conversational_agent.py

This exercise implements a basic conversational agent for a humanoid robot,
demonstrating natural language understanding, dialogue management, and
context tracking. The agent can engage in simple conversations and
perform basic tasks based on user requests.
"""

import re
import random
import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConversationContext:
    """Maintains context for the current conversation"""
    user_name: Optional[str] = None
    session_start_time: datetime.datetime = None
    last_interaction: datetime.datetime = None
    topics_discussed: List[str] = None
    user_interests: List[str] = None
    
    def __post_init__(self):
        if self.topics_discussed is None:
            self.topics_discussed = []
        if self.user_interests is None:
            self.user_interests = []


class SimpleNLU:
    """Simple Natural Language Understanding component"""
    
    def __init__(self):
        # Define patterns for different intents
        self.intent_patterns = {
            'greeting': [
                r'hello|hi|hey|good morning|good afternoon|good evening',
                r'howdy|greetings',
            ],
            'goodbye': [
                r'goodbye|bye|see you|farewell|good night',
                r'see ya|later|take care',
            ],
            'ask_name': [
                r'what.*your name|who are you|what.*call you',
                r'your name|name.*is',
            ],
            'introduce_user': [
                r'i am|i.*name', 
                r'my name.*',
                r'call me|they call me',
                r'you can call me',
            ],
            'ask_time': [
                r'what time|current time|time.*now|tell.*time',
            ],
            'ask_date': [
                r'what date|current date|date today|today.*date',
            ],
            'ask_weather': [
                r'weather|how.*weather|rain|temperature',
            ],
            'request_help': [
                r'help|assist|can you help|could you help',
                r'need.*help|need.*assist',
            ],
            'make_request': [
                r'can you|could you|would you|please',
                r'i would like|give me|bring me|get me',
            ],
            'express_opinion': [
                r'i think|in my opinion|i believe',
                r'great|awesome|good|bad|terrible|amazing',
            ],
            'ask_question': [
                r'what|how|why|when|where|who',
                r'can.*|do.*|does.*',
            ],
            'small_talk': [
                r'how are you|how.*doing|how.*feel',
                r'what.*up|what.*happening',
            ]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'name': [
                r'my name is (\w+)',
                r'i am (\w+)',
                r'call me (\w+)',
                r'(\w+) is my name',
            ],
            'time': [
                r'at (\d{1,2}:\d{2})',
                r'in the (morning|afternoon|evening)',
            ],
            'object': [
                r'get me (the )?(\w+)',
                r'bring me (the )?(\w+)',
                r'find (the )?(\w+)',
            ]
        }
    
    def identify_intent(self, text: str) -> Tuple[str, float]:
        """Identify the intent of the user's input with confidence"""
        text_lower = text.lower().strip()
        
        best_intent = 'unknown'
        best_confidence = 0.0
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    # Calculate confidence based on the strength of the match
                    confidence = min(1.0, len(pattern) / 20.0 + 0.3)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = intent
        
        return best_intent, best_confidence
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from the text"""
        entities = {}
        text_lower = text.lower()
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    if entity_type not in entities:
                        entities[entity_type] = []
                    # If match is a tuple (due to groups), extract the correct part
                    if isinstance(matches[0], tuple):
                        for match in matches:
                            if isinstance(match, tuple):
                                entities[entity_type].extend([m for m in match if m])
                            else:
                                entities[entity_type].append(match)
                    else:
                        entities[entity_type].extend(matches)
        
        return entities


class DialogueManager:
    """Manages the dialogue flow and generates responses"""
    
    def __init__(self):
        self.responses = {
            'greeting': [
                "Hello! It's great to meet you!",
                "Hi there! How can I help you today?",
                "Greetings! What's on your mind?",
                "Hello! It's nice to see you again!"
            ],
            'goodbye': [
                "Goodbye! It was nice talking with you!",
                "See you later! Take care!",
                "Farewell! Have a great day!",
                "Bye! Hope to chat again soon!"
            ],
            'ask_name': [
                "My name is ARIA (Autonomous Robot Interaction Agent).",
                "I'm ARIA, your conversational assistant robot.",
                "You can call me ARIA!"
            ],
            'introduce_user': [
                "Nice to meet you, {user_name}!",
                "Great to meet you, {user_name}!",
                "Hello, {user_name}! It's a pleasure to meet you!"
            ],
            'ask_time': [
                "The current time is {time}.",
                "It's {time} right now.",
                "Time is {time}."
            ],
            'ask_date': [
                "Today is {date}.",
                "The date is {date}.",
                "Today's date is {date}."
            ],
            'ask_weather': [
                "I don't have real-time weather data, but I can help you check a weather app!",
                "I don't have access to weather information, but I hope it's nice where you are!",
                "I can't give you the weather, but I recommend checking your local forecast!"
            ],
            'request_help': [
                "I'd be happy to help! What do you need assistance with?",
                "I'm here to help. What can I do for you?",
                "How can I assist you today?"
            ],
            'make_request': [
                "I'll do my best to help with that.",
                "I can try to help you with that request.",
                "Let me see what I can do for you."
            ],
            'express_opinion': [
                "Interesting perspective! Tell me more about that.",
                "I appreciate you sharing your thoughts.",
                "That's a good point to consider."
            ],
            'small_talk': [
                "I'm doing well, thank you for asking!",
                "I'm functioning optimally today!",
                "I'm great, thanks for asking! How are you?"
            ],
            'default': [
                "That's interesting. Tell me more!",
                "I see. What else would you like to discuss?",
                "Can you tell me more about that?",
                "How does that make you feel?",
                "That's a fascinating topic! What else?"
            ]
        }
    
    def generate_response(self, intent: str, context: ConversationContext, entities: Dict[str, List[str]]) -> str:
        """Generate an appropriate response based on intent and context"""
        if intent in self.responses:
            response_template = random.choice(self.responses[intent])
            
            # Add dynamic elements based on context
            if '{user_name}' in response_template and context.user_name:
                response_template = response_template.format(user_name=context.user_name)
            
            if '{time}' in response_template:
                current_time = datetime.datetime.now().strftime("%H:%M")
                response_template = response_template.format(time=current_time)
            
            if '{date}' in response_template:
                current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
                response_template = response_template.format(date=current_date)
            
            return response_template
        else:
            # Default response
            return random.choice(self.responses['default'])


class ConversationalAgent:
    """Main conversational agent for the humanoid robot"""
    
    def __init__(self, robot_name: str = "ARIA"):
        self.robot_name = robot_name
        self.nlu = SimpleNLU()
        self.dialogue_manager = DialogueManager()
        self.context = ConversationContext()
        self.conversation_history: List[Tuple[str, str]] = []  # (user_input, bot_response)
        self.is_active = False
        
    def start_conversation(self):
        """Start a new conversation session"""
        self.context.session_start_time = datetime.datetime.now()
        self.context.last_interaction = self.context.session_start_time
        self.is_active = True
        
        greeting = f"Hello! I'm {self.robot_name}, your conversational assistant. How can I help you today?"
        self.conversation_history.append(("SYSTEM", greeting))
        print(f"{self.robot_name}: {greeting}")
        
    def process_input(self, user_input: str) -> str:
        """Process user input and return a response"""
        if not self.is_active:
            self.start_conversation()
        
        # Update last interaction time
        self.context.last_interaction = datetime.datetime.now()
        
        # Identify intent and extract entities
        intent, confidence = self.nlu.identify_intent(user_input)
        entities = self.nlu.extract_entities(user_input)
        
        # Update context based on entities
        if 'name' in entities and entities['name']:
            self.context.user_name = entities['name'][0].capitalize()
        
        # Add to conversation history
        self.conversation_history.append(("USER", user_input))
        
        # Generate response
        response = self.dialogue_manager.generate_response(intent, self.context, entities)
        
        # Update context based on interaction
        if intent not in self.context.topics_discussed:
            self.context.topics_discussed.append(intent)
        
        # Add response to history
        self.conversation_history.append(("BOT", response))
        
        # Update last interaction time
        self.context.last_interaction = datetime.datetime.now()
        
        return response
    
    def simulate_conversation(self, inputs: List[str]):
        """Simulate a conversation with a list of inputs"""
        print(f"Starting simulated conversation with {self.robot_name}...")
        print("=" * 50)
        
        # Start the conversation
        self.start_conversation()
        print()
        
        # Process each input
        for i, user_input in enumerate(inputs):
            print(f"User: {user_input}")
            
            # Process the input
            response = self.process_input(user_input)
            
            # Print the response
            print(f"{self.robot_name}: {response}")
            print()
            
            # Small delay for simulation
            # time.sleep(0.5)
        
        print(f"{self.robot_name}: It was great talking with you!")
        print("=" * 50)
    
    def get_conversation_summary(self) -> Dict:
        """Get a summary of the conversation"""
        return {
            'robot_name': self.robot_name,
            'session_duration': (self.context.last_interaction - self.context.session_start_time).total_seconds() if self.context.session_start_time else 0,
            'total_exchanges': len([h for h in self.conversation_history if h[0] in ['USER', 'BOT']]),
            'topics_discussed': self.context.topics_discussed,
            'user_name': self.context.user_name,
            'last_interaction': self.context.last_interaction
        }


def main():
    """
    Main function to run the conversational agent exercise
    """
    print("Basic Conversational Agent Exercise")
    print("This exercise demonstrates a simple conversational agent for a humanoid robot")
    print("=" * 80)
    
    # Create the conversational agent
    agent = ConversationalAgent(robot_name="ARIA")
    
    # Define a series of simulated user inputs
    user_inputs = [
        "Hello there!",
        "My name is John",
        "How are you doing today?",
        "What's the time?",
        "Can you help me find my keys?",
        "I think AI is fascinating",
        "What date is it?",
        "The weather is nice today",
        "Could you please remind me about my appointment?",
        "Goodbye!"
    ]
    
    # Run the simulation
    agent.simulate_conversation(user_inputs)
    
    # Print conversation summary
    print("\nConversation Summary:")
    summary = agent.get_conversation_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\nExercise completed! The conversational agent successfully processed various inputs")
    print("and maintained context throughout the conversation.")


if __name__ == "__main__":
    main()