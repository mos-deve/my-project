"""
Talker - Conversational AI that generates text word by word.
Uses Markov chains to predict next word based on context.
"""

from __future__ import annotations
import random
import re
from collections import defaultdict, Counter


class MarkovLM:
    """
    Markov chain language model.
    
    Predicts next word based on previous n words.
    This is actual word-by-word generation.
    """
    
    def __init__(self, order: int = 3):
        self.order = order
        self.transitions: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        self.start_tokens: list[tuple[str, ...]] = []
        self.total_transitions = 0
    
    def train(self, texts: list[str]):
        """Train on texts."""
        for text in texts:
            tokens = self._tokenize(text)
            if len(tokens) < self.order + 1:
                continue
            
            # Add start and end markers
            tokens = ['<START>'] * self.order + tokens + ['<END>']
            
            # Record start token sequence
            start = tuple(tokens[:self.order])
            self.start_tokens.append(start)
            
            # Record transitions
            for i in range(len(tokens) - self.order):
                context = tuple(tokens[i:i + self.order])
                next_token = tokens[i + self.order]
                self.transitions[context][next_token] += 1
                self.total_transitions += 1
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text."""
        text = text.lower()
        return re.findall(r'\b\w+\b|[.!?]', text)
    
    def generate(self, seed: str | None = None, max_length: int = 30) -> str:
        """Generate text word by word."""
        # Start with seed or random start
        if seed:
            tokens = self._tokenize(seed)
            if len(tokens) >= self.order:
                current = tuple(tokens[-self.order:])
            else:
                current = random.choice(self.start_tokens) if self.start_tokens else ('<START>',) * self.order
        else:
            current = random.choice(self.start_tokens) if self.start_tokens else ('<START>',) * self.order
        
        result = []
        
        # Generate word by word
        for _ in range(max_length):
            if current not in self.transitions:
                break
            
            # Get possible next words and their counts
            next_options = self.transitions[current]
            
            if not next_options:
                break
            
            # Sample next word based on frequency
            words = list(next_options.keys())
            counts = list(next_options.values())
            total = sum(counts)
            probs = [c / total for c in counts]
            
            next_word = random.choices(words, weights=probs, k=1)[0]
            
            # Stop at end marker or sentence end
            if next_word in ['<END>', '.', '!', '?']:
                if result:  # Don't end immediately
                    break
                continue
            
            result.append(next_word)
            
            # Update context
            current = tuple(list(current[1:]) + [next_word])
        
        return self._detokenize(result)
    
    def _detokenize(self, tokens: list[str]) -> str:
        """Convert tokens to text."""
        if not tokens:
            return ""
        
        result = tokens[0]
        for i in range(1, len(tokens)):
            if tokens[i] in '.!?,:;':
                result += tokens[i]
            else:
                result += ' ' + tokens[i]
        
        return result
    
    def predict_next(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Predict top-k next words."""
        tokens = self._tokenize(text)
        
        if len(tokens) < self.order:
            return []
        
        current = tuple(tokens[-self.order:])
        
        if current not in self.transitions:
            return []
        
        next_options = self.transitions[current]
        total = sum(next_options.values())
        
        predictions = []
        for word, count in next_options.most_common(top_k):
            prob = count / total
            predictions.append((word, prob))
        
        return predictions
    
    def get_stats(self) -> dict:
        """Get model statistics."""
        return {
            "order": self.order,
            "unique_contexts": len(self.transitions),
            "total_transitions": self.total_transitions,
        }


class Talker:
    """
    Conversational AI that generates text word by word.
    
    Uses a Markov chain to predict each next word based on context.
    This is actual generation - each word is chosen based on probability.
    """
    
    def __init__(self, order: int = 3):
        self.model = MarkovLM(order=order)
        self.conversation_history: list[dict] = []
        self._train_default()
    
    def _train_default(self):
        """Train with comprehensive knowledge."""
        training_texts = [
            # Programming
            "Programming is the process of creating instructions for computers.",
            "Code is written in programming languages like Python and JavaScript.",
            "A function is a reusable block of code that performs a specific task.",
            "Variables store data values that can be used and modified in a program.",
            "Algorithms are step by step procedures for solving computational problems.",
            "Debugging is the process of finding and fixing errors in computer code.",
            "Good code is readable, maintainable, efficient, and well documented.",
            "Data structures organize and store data in efficient ways for access.",
            "Recursion is when a function calls itself to solve smaller subproblems.",
            "Loops repeat a block of code multiple times until a condition is met.",
            "Object oriented programming organizes code into classes and objects.",
            "Software development involves designing, coding, testing, and deploying.",
            "Testing ensures that code works correctly and meets requirements.",
            "Version control tracks changes to code over time using tools like git.",
            
            # AI
            "Artificial intelligence is the simulation of human intelligence by machines.",
            "Machine learning allows computers to learn patterns from data automatically.",
            "Neural networks are computing systems inspired by the structure of the brain.",
            "Deep learning uses many layers of neural networks to learn complex patterns.",
            "Training is the process of teaching a machine learning model using data.",
            "Inference is using a trained model to make predictions on new data.",
            "Natural language processing helps computers understand and generate text.",
            "Reinforcement learning trains agents by rewarding good actions and penalizing bad ones.",
            "Computer vision enables machines to interpret and understand visual information.",
            "Transfer learning reuses a trained model for a different but related task.",
            
            # Mathematics
            "Mathematics is the abstract science of number, quantity, and space.",
            "Numbers and symbols are used to express mathematical ideas and relationships.",
            "Algebra uses letters and symbols to represent unknown values in equations.",
            "Calculus is the mathematical study of continuous change and motion.",
            "Statistics is the science of collecting, analyzing, and interpreting data.",
            "Probability is the measure of how likely an event is to occur.",
            "Geometry is the branch of mathematics dealing with shapes and spaces.",
            "Arithmetic is the branch of mathematics dealing with basic number operations.",
            "Linear algebra studies vectors, matrices, and linear transformations.",
            "Number theory studies the properties and relationships of integers.",
            
            # Philosophy
            "Philosophy explores fundamental questions about existence, knowledge, and values.",
            "Ethics is the branch of philosophy that studies moral right and wrong.",
            "Consciousness is the state of being aware of oneself and surroundings.",
            "Free will is the ability to make choices that are not determined by prior causes.",
            "Logic is the study of valid reasoning and argumentation principles.",
            "Epistemology is the study of the nature and limits of knowledge.",
            "Metaphysics studies the fundamental nature of reality and existence.",
            
            # Technology
            "Computers are electronic devices that process data according to instructions.",
            "The internet is a global network connecting millions of computers worldwide.",
            "Technology is the application of scientific knowledge for practical purposes.",
            "Software consists of programs and data that tell hardware what to do.",
            "Hardware refers to the physical components of a computer system.",
            "Cloud computing provides on demand access to computing resources over the internet.",
            "Cybersecurity protects computer systems and networks from digital attacks.",
            "Databases organize and store large amounts of structured data efficiently.",
            
            # Learning
            "Learning is the process of acquiring new knowledge, skills, and behaviors.",
            "Practice and repetition strengthen neural pathways and improve performance.",
            "Feedback provides information about how well something is working.",
            "Curiosity is the desire to learn and explore new things.",
            "Education is the process of facilitating learning and acquiring knowledge.",
            "Memory is the faculty of the brain by which information is stored and retrieved.",
            "Understanding comes from connecting new information to existing knowledge.",
            "Mastery requires sustained practice and deliberate effort over time.",
            
            # ANDROM
            "I am ANDROM, a self improving artificial intelligence system.",
            "I use computational units to process information and generate responses.",
            "My rule engine applies logical rules to make decisions and solve problems.",
            "I can learn from conversations and improve my responses over time.",
            "I generate text word by word using a Markov chain model.",
            "I predict the next word based on the previous context of words.",
            "I am designed to be helpful, informative, and conversational.",
            "I combine deterministic rules with probabilistic reasoning methods.",
            
            # Greetings
            "Hello! How can I help you today with your question?",
            "Hi there! What would you like to know about the topic?",
            "Greetings! I am ready to assist you with information.",
            "Hey! What is on your mind that I can help with?",
            "Goodbye! Have a great day and come back anytime!",
            "See you later! Take care and feel free to return!",
            "Bye! Feel free to come back if you have more questions!",
            
            # Conversational
            "I can help you with that question about the topic.",
            "Let me explain that concept in more detail for you.",
            "That is a great observation about the subject matter.",
            "Here is what I know about that particular topic.",
            "I am happy to assist you with that request.",
            "That is an interesting perspective to consider carefully.",
            "I agree with that point of view on the matter.",
            "Let me think about that for a moment before answering.",
            "That makes sense when you consider the full context.",
            "I understand what you are asking about clearly.",
        ]
        
        print("Training Markov chain model...")
        self.model.train(training_texts)
        print(f"Model trained: {self.model.get_stats()}")
    
    def respond(self, user_input: str) -> str:
        """
        Generate a response word by word.
        
        Each word is predicted based on previous words,
        like how humans produce speech.
        """
        self.conversation_history.append({"role": "user", "text": user_input})
        
        # Generate response
        response = self._generate_response(user_input)
        
        # Clean up
        response = self._clean_response(response)
        
        self.conversation_history.append({"role": "assistant", "text": response})
        return response
    
    def _generate_response(self, prompt: str) -> str:
        """Generate a response to the prompt."""
        prompt_lower = prompt.lower()
        
        # Choose seed based on topic
        if any(g in prompt_lower for g in ['hello', 'hi ', 'hey']):
            seed = "Hello"
        elif any(f in prompt_lower for f in ['bye', 'goodbye', 'see you']):
            seed = "Goodbye"
        elif 'programming' in prompt_lower or 'code' in prompt_lower:
            seed = "Programming is"
        elif 'ai' in prompt_lower or 'artificial' in prompt_lower:
            seed = "Artificial intelligence"
        elif 'math' in prompt_lower:
            seed = "Mathematics is"
        elif 'androm' in prompt_lower or ('you' in prompt_lower and 'how' in prompt_lower):
            seed = "I am"
        elif 'learn' in prompt_lower:
            seed = "Learning is"
        elif 'philosophy' in prompt_lower or 'consciousness' in prompt_lower:
            seed = "Philosophy explores"
        elif 'science' in prompt_lower:
            seed = "Science is"
        elif 'technology' in prompt_lower or 'computer' in prompt_lower:
            seed = "Computers are"
        elif '?' in prompt:
            seed = "That is"
        else:
            seed = "I can"
        
        # Generate word by word
        response = self.model.generate(seed, max_length=20)
        
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
    
    def train(self, texts: list[str]):
        """Train on additional texts."""
        self.model.train(texts)
    
    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []
    
    def get_stats(self) -> dict:
        """Get model statistics."""
        stats = self.model.get_stats()
        stats["conversation_turns"] = len(self.conversation_history)
        return stats
