"""
Talker - Probabilistic conversational text generator for ANDROM.
Learns patterns from text and generates coherent responses probabilistically.
"""

from __future__ import annotations
import random
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NGramModel:
    """N-gram language model for text generation."""
    n: int = 3
    counts: dict[tuple, Counter] = field(default_factory=lambda: defaultdict(Counter))
    total_contexts: dict[tuple, int] = field(default_factory=lambda: defaultdict(int))
    
    def train(self, text: str):
        """Train on text."""
        tokens = self._tokenize(text)
        if len(tokens) < self.n:
            return
        
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i + self.n - 1])
            next_token = tokens[i + self.n - 1]
            self.counts[context][next_token] += 1
            self.total_contexts[context] += 1
    
    def train_many(self, texts: list[str]):
        """Train on multiple texts."""
        for text in texts:
            self.train(text)
    
    def generate(self, max_length: int = 50, seed: str | list | None = None) -> str:
        """Generate text probabilistically."""
        if not self.counts:
            return ""
        
        # Start with seed or random context
        if seed:
            if isinstance(seed, list):
                tokens = seed
            else:
                tokens = self._tokenize(seed)
            if len(tokens) >= self.n - 1:
                current = tuple(tokens[-(self.n - 1):])
            else:
                current = random.choice(list(self.counts.keys()))
        else:
            current = random.choice(list(self.counts.keys()))
        
        result = list(current)
        
        for _ in range(max_length):
            if current not in self.counts:
                break
            
            # Probabilistic next token selection
            next_token = self._sample_next(current)
            if next_token is None:
                break
            
            result.append(next_token)
            current = tuple(result[-(self.n - 1):])
            
            # Stop at sentence end
            if next_token in '.!?':
                break
        
        return self._detokenize(result)
    
    def _sample_next(self, context: tuple) -> str | None:
        """Sample next token from probability distribution."""
        if context not in self.counts:
            return None
        
        counter = self.counts[context]
        total = self.total_contexts[context]
        
        # Create probability distribution
        tokens = list(counter.keys())
        weights = [counter[t] / total for t in tokens]
        
        # Sample
        return random.choices(tokens, weights=weights, k=1)[0]
    
    def get_probability(self, context: tuple, token: str) -> float:
        """Get probability of token given context."""
        if context not in self.counts:
            return 0.0
        counter = self.counts[context]
        total = self.total_contexts[context]
        return counter.get(token, 0) / total if total > 0 else 0.0
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text."""
        # Split on whitespace and punctuation
        tokens = re.findall(r'\S+|[.!?]', text)
        return tokens
    
    def _detokenize(self, tokens: list[str]) -> str:
        """Convert tokens back to text."""
        if not tokens:
            return ""
        
        result = tokens[0]
        for i in range(1, len(tokens)):
            # Add space unless punctuation
            if tokens[i] in '.!?,:;':
                result += tokens[i]
            else:
                result += ' ' + tokens[i]
        
        return result


@dataclass
class CharModel:
    """Character-level language model for fine-grained generation."""
    counts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    totals: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    context_size: int = 5
    
    def train(self, text: str):
        """Train on text."""
        for i in range(len(text) - self.context_size):
            context = text[i:i + self.context_size]
            next_char = text[i + self.context_size]
            self.counts[context][next_char] += 1
            self.totals[context] += 1
    
    def train_many(self, texts: list[str]):
        """Train on multiple texts."""
        for text in texts:
            self.train(text)
    
    def generate(self, max_length: int = 200, seed: str | None = None) -> str:
        """Generate text character by character."""
        if not self.counts:
            return ""
        
        # Start with seed
        if seed and len(seed) >= self.context_size:
            current = seed[-self.context_size:]
            result = seed
        else:
            current = random.choice(list(self.counts.keys()))
            result = current
        
        for _ in range(max_length):
            if current not in self.counts:
                break
            
            next_char = self._sample_char(current)
            if next_char is None:
                break
            
            result += next_char
            current = result[-self.context_size:]
            
            # Stop at sentence end
            if next_char in '.!?\n' and len(result) > 20:
                break
        
        return result
    
    def _sample_char(self, context: str) -> str | None:
        """Sample next character from probability distribution."""
        if context not in self.counts:
            return None
        
        counter = self.counts[context]
        total = self.totals[context]
        
        chars = list(counter.keys())
        weights = [counter[c] / total for c in chars]
        
        return random.choices(chars, weights=weights, k=1)[0]


class Talker:
    """
    Probabilistic conversational text generator.
    
    Uses multiple models:
    - N-gram for word-level generation
    - Character-level for fine control
    - Learned patterns from training data
    """
    
    def __init__(self, ngram_size: int = 3):
        self.word_model = NGramModel(n=ngram_size)
        self.char_model = CharModel(context_size=5)
        self.conversation_history: list[dict] = []
        self.response_buffer: list[str] = []
        
        # Training corpus
        self._training_texts: list[str] = []
        
        # Topic-response pairs for coherent responses
        self._topic_responses: dict[str, list[str]] = {}
        
        # Initialize with default knowledge
        self._init_training_data()
        self._train()
    
    def _init_training_data(self):
        """Initialize with training data."""
        # Complete conversations for learning patterns
        self._training_texts = [
            "Hello! How are you doing today? I am doing well, thank you for asking.",
            "What is your name? My name is ANDROM. I am a self-improving AI system.",
            "How do you work? I use thousands of mathematical units to process information.",
            "What can you do? I can solve problems, generate code, and have conversations.",
            "Tell me about yourself. I am an artificial intelligence that learns and improves.",
            
            "Programming is the art of telling a computer what to do. Code is written in languages like Python.",
            "A function takes inputs and returns outputs. Variables store data. Loops repeat operations.",
            "Algorithms are step by step procedures for solving problems. Data structures organize information.",
            "Debugging is finding and fixing errors in code. Testing ensures code works correctly.",
            "Good code is readable, maintainable, and efficient. Comments explain what code does.",
            
            "Artificial intelligence is the simulation of human intelligence by machines.",
            "Machine learning allows computers to learn from data without being explicitly programmed.",
            "Neural networks are inspired by the human brain. They have layers of connected nodes.",
            "Training is the process of teaching a model using data. Inference is using the model.",
            "Deep learning uses many layers to learn complex patterns in data.",
            
            "Mathematics is the language of science and engineering. Numbers and symbols express ideas.",
            "Arithmetic includes addition, subtraction, multiplication, and division.",
            "Algebra uses letters to represent unknown values. Equations show relationships.",
            "Geometry studies shapes and spaces. Calculus studies change and motion.",
            "Statistics analyzes data and finds patterns. Probability measures uncertainty.",
            
            "What is consciousness? How do we know what is real? These are philosophical questions.",
            "Ethics studies right and wrong. Logic studies valid reasoning.",
            "Free will versus determinism is a classic philosophical debate.",
            "The meaning of life is a question philosophers have explored for centuries.",
            
            "Computers process information using circuits and software. The internet connects computers.",
            "Technology changes how we live and work. Innovation drives progress.",
            "Software runs on hardware. Programs are written by programmers.",
            "Data is information stored in computers. Databases organize large amounts of data.",
            
            "I can improve myself by learning from experience. Each interaction teaches me something.",
            "My code can be optimized to run faster and use less memory.",
            "I analyze my own source code to find ways to improve.",
            "Learning never stops. There is always more to discover and understand.",
            
            "ANDROM stands for Adaptive Network of Deterministic Rule-based Operations and Mathematics.",
            "I have thousands of computational units that process information in parallel.",
            "My rule engine applies logical rules to make decisions.",
            "I can generate code, solve problems, and have conversations.",
            "I use genetic algorithms to evolve better solutions over time.",
        ]
        
        # Topic-specific response templates
        self._topic_responses = {
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! What would you like to know?",
                "Greetings! I am ready to assist you.",
            ],
            "farewell": [
                "Goodbye! Have a great day!",
                "See you later! Take care!",
                "Bye! Feel free to come back anytime.",
            ],
            "programming": [
                "Programming is about solving problems with code. I can help with algorithms and data structures.",
                "Code is written in languages like Python. Functions take inputs and return outputs.",
                "Good code is readable and efficient. Debugging is finding and fixing errors.",
            ],
            "ai": [
                "Artificial intelligence simulates human intelligence in machines. I am an example of AI.",
                "Machine learning lets computers learn from data. Neural networks are inspired by the brain.",
                "AI has many applications. I use AI to learn and improve myself.",
            ],
            "math": [
                "Mathematics is the language of science. It helps us understand patterns and relationships.",
                "Math includes arithmetic, algebra, geometry, and calculus. Numbers and symbols express ideas.",
                "Statistics analyzes data. Probability measures uncertainty. Math is fundamental to computing.",
            ],
            "self": [
                "I am ANDROM, a self-improving AI system. I learn from every interaction.",
                "I have thousands of computational units. My rule engine helps me make decisions.",
                "I can improve my own code. I analyze myself and find ways to optimize.",
            ],
            "general": [
                "That is an interesting topic. Let me think about it.",
                "I see. That is something worth considering.",
                "Good question. I am processing that information.",
            ],
        }
    
    def _train(self):
        """Train all models."""
        self.word_model.train_many(self._training_texts)
        self.char_model.train_many(self._training_texts)
    
    def add_training_text(self, text: str):
        """Add new training text and retrain."""
        self._training_texts.append(text)
        self.word_model.train(text)
        self.char_model.train(text)
    
    def respond(self, user_input: str) -> str:
        """
        Generate a response to user input.
        
        Uses probabilistic generation based on learned patterns.
        """
        self.conversation_history.append({"role": "user", "text": user_input})
        
        # Detect topic
        topic = self._detect_topic(user_input)
        
        # Generate response
        response = self._generate_coherent_response(topic, user_input)
        
        # Clean up response
        response = self._clean_response(response)
        
        self.conversation_history.append({"role": "assistant", "text": response})
        return response
    
    def _detect_topic(self, text: str) -> str:
        """Detect the topic of user input."""
        text_lower = text.lower()
        
        topic_keywords = {
            "greeting": ["hello", "hi ", "hey", "greetings", "howdy", "good morning", "good afternoon"],
            "farewell": ["bye", "goodbye", "see you", "farewell", "later", "take care"],
            "programming": ["code", "program", "function", "variable", "python", "javascript", "algorithm", "debug"],
            "ai": ["ai", "artificial intelligence", "machine learning", "neural", "model", "training", "deep learning"],
            "math": ["math", "number", "calculate", "equation", "formula", "geometry", "algebra", "statistics"],
            "philosophy": ["philosophy", "ethics", "moral", "existence", "meaning", "consciousness", "free will"],
            "technology": ["technology", "computer", "software", "hardware", "internet", "digital", "data"],
            "self": ["you", "yourself", "androm", "how do you", "what are you", "self-improve", "evolve"],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return topic
        
        return "general"
    
    def _generate_coherent_response(self, topic: str, user_input: str) -> str:
        """Generate a coherent response based on topic."""
        # 70% chance to use topic-specific response
        if random.random() < 0.7 and topic in self._topic_responses:
            base_response = random.choice(self._topic_responses[topic])
            
            # Try to extend with n-gram model
            if len(base_response.split()) < 10:
                extension = self.word_model.generate(max_length=15, seed=base_response.split()[-2:])
                if extension and len(extension.split()) > 3:
                    base_response = base_response + " " + extension
            
            return base_response
        
        # 30% chance to generate purely probabilistically
        # Find relevant seed from training data
        seed = None
        for text in self._training_texts:
            text_lower = text.lower()
            words = user_input.lower().split()
            for word in words:
                if len(word) > 3 and word in text_lower:
                    # Get the sentence containing this word
                    sentences = re.split(r'[.!?]', text)
                    for sent in sentences:
                        if word in sent.lower():
                            seed = sent.strip()
                            break
                    if seed:
                        break
            if seed:
                break
        
        if seed:
            response = self.word_model.generate(max_length=25, seed=seed)
            if response and len(response.split()) >= 4:
                return response
        
        # Fallback to topic response
        if topic in self._topic_responses:
            return random.choice(self._topic_responses[topic])
        
        return random.choice(self._topic_responses["general"])
    
    def _clean_response(self, text: str) -> str:
        """Clean up generated response."""
        if not text:
            return "I am thinking about that."
        
        # Capitalize first letter
        text = text.strip()
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
    
    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []
    
    def get_stats(self) -> dict:
        """Get model statistics."""
        return {
            "word_model_contexts": len(self.word_model.counts),
            "char_model_contexts": len(self.char_model.counts),
            "training_texts": len(self._training_texts),
            "conversation_turns": len(self.conversation_history),
            "topics": list(self._topic_responses.keys()),
        }
