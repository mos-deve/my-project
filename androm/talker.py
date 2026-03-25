"""
Talker - Conversational text generator using advanced MemNet.
Smart, relevant, context-aware responses.
"""

from __future__ import annotations
import random
from androm.memnet import MemNet


class Talker:
    """
    Smart conversational AI using MemNet.
    
    Features:
    - Semantic understanding of questions
    - Topic-aware response selection
    - Multi-signal relevance scoring
    - Context tracking
    - Continuous learning
    """
    
    def __init__(self, memory_size: int = 10000):
        self.model = MemNet(memory_size=memory_size)
        self.conversation_history: list[dict] = []
        self._train_comprehensive()
    
    def _train_comprehensive(self):
        """Train with comprehensive knowledge base."""
        # Question-answer pairs organized by topic
        knowledge = {
            # Greetings
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! I'm ANDROM, ready to assist you.",
                "Greetings! What would you like to know?",
                "Hey! Great to meet you. What's on your mind?",
                "Goodbye! Have a great day!",
                "See you later! Take care!",
                "Bye! Feel free to come back anytime.",
            ],
            
            # Programming knowledge
            "programming": [
                "Programming is the process of creating instructions for computers. Code is written in languages like Python, JavaScript, and Rust.",
                "A function is a reusable block of code that takes inputs and returns outputs. Functions help organize and modularize code.",
                "Debugging is the process of finding and fixing errors in code. Good programmers use debuggers, print statements, and logical reasoning.",
                "An algorithm is a step-by-step procedure for solving a problem. Good algorithms are efficient, correct, and easy to understand.",
                "Variables store data values. They have names and types. Good variable names describe what the variable contains.",
                "Object-oriented programming organizes code into classes and objects. Classes define behavior, objects are instances.",
                "Data structures organize and store data efficiently. Common structures include arrays, lists, trees, and hash tables.",
                "Recursion is when a function calls itself. It's useful for problems that can be broken into smaller similar subproblems.",
            ],
            
            # AI knowledge
            "ai": [
                "Artificial intelligence is the simulation of human intelligence by machines. It includes learning, reasoning, and problem-solving.",
                "Machine learning is a subset of AI where computers learn from data without being explicitly programmed for every scenario.",
                "Neural networks are computing systems inspired by biological brains. They have layers of interconnected nodes that process information.",
                "Deep learning uses neural networks with many layers to learn complex patterns. It's used for image recognition, language processing, and more.",
                "Training is the process of teaching a model using data. The model adjusts its parameters to minimize errors on the training data.",
                "Natural language processing (NLP) helps computers understand and generate human language. It powers chatbots, translation, and analysis.",
                "Reinforcement learning trains agents to make decisions by rewarding good actions and penalizing bad ones.",
                "Transfer learning reuses a model trained on one task for a different but related task, saving time and data.",
            ],
            
            # Math knowledge
            "math": [
                "Mathematics is the study of numbers, structures, and patterns. It's the foundation of science and engineering.",
                "Algebra uses symbols and letters to represent unknown values. Equations express relationships between quantities.",
                "Calculus studies continuous change. Differential calculus finds rates of change, integral calculus finds areas and volumes.",
                "Statistics collects, analyzes, and interprets data. It helps make decisions under uncertainty.",
                "Probability measures the likelihood of events. It ranges from 0 (impossible) to 1 (certain).",
                "Geometry studies shapes, sizes, and properties of space. Euclidean geometry deals with flat spaces.",
                "Linear algebra studies vectors and matrices. It's fundamental to computer graphics and machine learning.",
                "Number theory studies properties of integers. It includes prime numbers, divisibility, and congruences.",
            ],
            
            # Philosophy
            "philosophy": [
                "Philosophy explores fundamental questions about existence, knowledge, values, reason, and language.",
                "Ethics studies right and wrong, good and bad. It asks how we should live and treat others.",
                "Epistemology studies knowledge. It asks what knowledge is, how we acquire it, and what we can know.",
                "Consciousness is the awareness of ourselves and our surroundings. Its nature is one of philosophy's deepest questions.",
                "Free will is the ability to choose our actions. Philosophers debate whether it exists or if all events are determined.",
                "Logic studies valid reasoning. It provides tools for constructing and evaluating arguments.",
                "Existentialism emphasizes individual existence, freedom, and choice. It asks how we create meaning in an indifferent universe.",
            ],
            
            # Technology
            "technology": [
                "Technology applies scientific knowledge for practical purposes. It shapes how we live, work, and communicate.",
                "Computers process information using electronic circuits. They execute programs that perform calculations and operations.",
                "The internet connects billions of computers worldwide. It enables instant communication, information sharing, and collaboration.",
                "Software consists of programs and data that tell hardware what to do. Good software is reliable, efficient, and user-friendly.",
                "Artificial intelligence is transforming technology. It powers recommendation systems, autonomous vehicles, and medical diagnosis.",
                "Cybersecurity protects systems and data from attacks. It includes encryption, authentication, and intrusion detection.",
                "Cloud computing provides on-demand access to computing resources. It enables scalable, flexible, and cost-effective solutions.",
            ],
            
            # ANDROM specific
            "androm": [
                "I am ANDROM, an Adaptive Network of Deterministic Rule-based Operations and Mathematics.",
                "I use thousands of computational units to process information in parallel.",
                "My rule engine applies logical rules to make decisions and solve problems.",
                "I can learn from conversations and improve my responses over time.",
                "I use genetic algorithms to evolve better solutions to problems.",
                "My memory system stores knowledge and retrieves relevant information for each question.",
                "I analyze code, generate solutions, and optimize my own performance.",
                "I combine deterministic rules with probabilistic reasoning for smart decisions.",
            ],
            
            # Learning and self-improvement
            "learning": [
                "Learning is acquiring new knowledge, skills, or behaviors through study, experience, or teaching.",
                "Practice and repetition strengthen neural pathways, making skills more automatic over time.",
                "Feedback helps us learn by showing what works and what doesn't. Constructive feedback accelerates improvement.",
                "Curiosity drives learning. Asking questions and exploring leads to deeper understanding.",
                "Growth mindset believes abilities can be developed through effort. It encourages persistence and learning from failure.",
                "Deliberate practice focuses on specific aspects of performance with immediate feedback.",
                "Metacognition is thinking about thinking. It helps us understand and improve our learning processes.",
            ],
            
            # Science
            "science": [
                "Science is the systematic study of the natural world through observation and experiment.",
                "The scientific method involves forming hypotheses, testing them, and drawing conclusions from evidence.",
                "Physics studies matter, energy, and their interactions. It explains everything from atoms to galaxies.",
                "Biology studies living organisms. It includes genetics, evolution, ecology, and physiology.",
                "Chemistry studies substances and their transformations. It explains how matter changes and bonds form.",
            ],
        }
        
        # Store all knowledge
        for topic, texts in knowledge.items():
            for text in texts:
                self.model.memory_bank.store(text, memory_type="response", topic=topic)
    
    def respond(self, user_input: str) -> str:
        """
        Generate a smart, relevant response.
        """
        self.conversation_history.append({"role": "user", "text": user_input})
        
        # Generate response
        response = self.model.generate(user_input)
        
        # Clean up
        response = self._clean_response(response)
        
        # Learn from this exchange
        self.model.memory_bank.store(user_input, memory_type="question")
        self.model.memory_bank.store(response, memory_type="response")
        
        self.conversation_history.append({"role": "assistant", "text": response})
        return response
    
    def _clean_response(self, text: str) -> str:
        """Clean up generated response."""
        if not text:
            return "I'm thinking about that. Could you tell me more?"
        
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
    
    def learn(self, text: str, topic: str | None = None):
        """Learn new information."""
        self.model.memory_bank.store(text, memory_type="response", topic=topic)
    
    def learn_many(self, texts: list[str], topic: str | None = None):
        """Learn multiple pieces of information."""
        for text in texts:
            self.learn(text, topic)
    
    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []
    
    def get_stats(self) -> dict:
        """Get model statistics."""
        stats = self.model.get_stats()
        stats["conversation_turns"] = len(self.conversation_history)
        return stats
