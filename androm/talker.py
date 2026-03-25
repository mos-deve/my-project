"""
Talker - Conversational text generator using MemNet.
Uses memory-augmented neural network for coherent text generation.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from androm.memnet import MemNet


class Talker:
    """
    Conversational text generator using MemNet architecture.
    
    Uses memory-augmented neural network instead of:
    - Transformers (no quadratic attention)
    - N-grams (better context understanding)
    - RNNs (can access any memory)
    
    Features:
    - Learns from conversations
    - Retrieves relevant memories for context
    - Generates coherent responses
    - Memory grows with use
    """
    
    def __init__(self, memory_size: int = 10000):
        self.model = MemNet(memory_size=memory_size)
        self.conversation_history: list[dict] = []
        self._train_default()
    
    def _train_default(self):
        """Train with default knowledge - responses only."""
        # Store question-answer pairs
        qa_pairs = [
            ("Hello!", "Hello! How can I help you today?"),
            ("Hi", "Hi there! What would you like to know?"),
            ("Hey", "Hey! What's on your mind?"),
            
            ("What is programming?", "Programming is about solving problems with code. Code is written in languages like Python."),
            ("How do functions work?", "Functions take inputs and return outputs. They organize code into reusable pieces."),
            ("What is debugging?", "Debugging is finding and fixing errors in code. Testing ensures code works correctly."),
            ("What makes good code?", "Good code is readable, maintainable, and efficient. Comments explain what code does."),
            
            ("What is AI?", "Artificial intelligence simulates human intelligence in machines. I am an example of AI."),
            ("What is machine learning?", "Machine learning lets computers learn from data without being explicitly programmed."),
            ("What are neural networks?", "Neural networks are inspired by the human brain. They have layers of connected nodes."),
            ("How does training work?", "Training is the process of teaching a model using data. Inference is using the model."),
            
            ("What is mathematics?", "Mathematics is the language of science. Numbers and symbols express ideas."),
            ("What is algebra?", "Algebra uses letters to represent unknown values. Equations show relationships."),
            ("What is statistics?", "Statistics analyzes data and finds patterns. Probability measures uncertainty."),
            
            ("What is ANDROM?", "ANDROM stands for Adaptive Network of Deterministic Rule-based Operations and Mathematics."),
            ("How do you work?", "I use thousands of computational units to process information. My rule engine helps me make decisions."),
            ("How do you improve?", "I can improve myself by learning from experience. Each interaction teaches me something."),
            ("What can you do?", "I can solve problems, generate code, and have conversations. I use genetic algorithms to evolve better solutions."),
            
            ("What is philosophy?", "Philosophy explores fundamental questions about existence. Ethics studies right and wrong."),
            ("What is consciousness?", "Consciousness is a deep philosophical question. How do we know what is real?"),
            
            ("What is technology?", "Computers process information using circuits and software. Technology changes how we live and work."),
            ("How does the internet work?", "The internet connects computers globally. It enables communication and information sharing."),
            
            ("Goodbye!", "Goodbye! Have a great day!"),
            ("Bye", "See you later! Take care!"),
        ]
        
        # Store responses in memory
        for question, answer in qa_pairs:
            self.model.memory_bank.store(question, memory_type="question")
            self.model.memory_bank.store(answer, memory_type="response")
    
    def respond(self, user_input: str) -> str:
        """
        Generate a response to user input.
        
        Uses MemNet to:
        1. Encode the input
        2. Retrieve relevant memories
        3. Generate coherent response
        """
        self.conversation_history.append({"role": "user", "text": user_input})
        
        # Generate response using MemNet
        response = self.model.generate(user_input, max_length=50, temperature=0.7)
        
        # Clean up response
        response = self._clean_response(response)
        
        # Store in memory for learning
        self.model.memory_bank.store(user_input, memory_type="question")
        self.model.memory_bank.store(response, memory_type="response")
        
        self.conversation_history.append({"role": "assistant", "text": response})
        return response
    
    def _clean_response(self, text: str) -> str:
        """Clean up generated response."""
        if not text:
            return "I am thinking about that."
        
        text = text.strip()
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        # Ensure ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        # Remove duplicate spaces
        text = ' '.join(text.split())
        
        return text
    
    def chat(self, user_input: str) -> str:
        """Alias for respond."""
        return self.respond(user_input)
    
    def learn(self, text: str):
        """Learn from text."""
        self.model.memory_bank.store(text, memory_type="response")
    
    def learn_many(self, texts: list[str]):
        """Learn from multiple texts."""
        for text in texts:
            self.learn(text)
    
    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []
    
    def get_stats(self) -> dict:
        """Get model statistics."""
        stats = self.model.get_stats()
        stats["conversation_turns"] = len(self.conversation_history)
        return stats
