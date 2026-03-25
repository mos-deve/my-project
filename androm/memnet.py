"""
MemNet - Advanced Memory-Augmented Neural Network for text generation.
Smart retrieval with semantic understanding and contextual relevance.
"""

from __future__ import annotations
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class Memory:
    """A single memory entry with rich metadata."""
    key: np.ndarray
    value: str
    tokens: list[str] = field(default_factory=list)
    memory_type: str = "response"
    topic: str = "general"
    access_count: int = 0
    created_at: int = 0
    relevance_score: float = 0.0
    
    def similarity(self, query: np.ndarray) -> float:
        """Cosine similarity between query and key."""
        dot = np.dot(self.key, query)
        norm = np.linalg.norm(self.key) * np.linalg.norm(query)
        if norm == 0:
            return 0.0
        return dot / norm


class SmartEncoder:
    """Advanced text encoder with TF-IDF weighting and semantic features."""
    
    def __init__(self, vocab_size: int = 5000, embed_dim: int = 256):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # TF-IDF components
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.total_docs: int = 0
        self.idf: dict[str, float] = {}
        
        # Word embeddings (pre-trained style)
        self.word_vectors: dict[str, np.ndarray] = {}
        
        # Token mapping
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        
        # Topic keywords for classification
        self.topic_keywords: dict[str, set[str]] = {
            "greeting": {"hello", "hi", "hey", "greetings", "goodbye", "bye", "see you"},
            "programming": {"code", "program", "function", "variable", "python", "algorithm", "debug", "software", "recursion", "object", "class", "loop", "array"},
            "ai": {"ai", "artificial", "intelligence", "machine", "learning", "neural", "network", "model", "training", "deep", "nlp", "reinforcement"},
            "math": {"math", "mathematics", "number", "calculate", "equation", "formula", "geometry", "algebra", "statistics", "probability", "calculus"},
            "philosophy": {"philosophy", "ethics", "moral", "existence", "meaning", "consciousness", "free", "will", "epistemology", "logic"},
            "technology": {"technology", "computer", "software", "hardware", "internet", "digital", "data", "cloud", "cybersecurity"},
            "androm": {"androm", "yourself", "self-improve", "evolve", "you", "your"},
            "learning": {"learn", "learning", "practice", "feedback", "curiosity", "growth", "improve"},
            "science": {"science", "physics", "biology", "chemistry", "experiment", "hypothesis"},
        }
        
        self._init_vocabulary()
        self._init_word_vectors()
    
    def _init_vocabulary(self):
        """Initialize comprehensive vocabulary."""
        specials = ['<PAD>', '<UNK>', '<START>', '<END>']
        for i, token in enumerate(specials):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        # Comprehensive word list
        words = [
            # Common words
            'the', 'a', 'an', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may',
            'not', 'no', 'yes', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'and', 'or', 'but', 'if', 'when', 'because', 'while', 'although',
            'that', 'what', 'which', 'who', 'how', 'where', 'why', 'when',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up',
            'about', 'into', 'through', 'during', 'before', 'after', 'above',
            # Verbs
            'make', 'take', 'come', 'go', 'get', 'know', 'think', 'see', 'want',
            'use', 'find', 'give', 'tell', 'work', 'call', 'try', 'need', 'feel',
            'become', 'leave', 'put', 'mean', 'keep', 'let', 'begin', 'show', 'hear',
            'play', 'run', 'move', 'live', 'believe', 'help', 'learn', 'create', 'solve',
            'improve', 'process', 'generate', 'analyze', 'understand', 'explain',
            # Nouns
            'time', 'year', 'people', 'way', 'day', 'world', 'life', 'hand', 'part',
            'place', 'case', 'week', 'company', 'system', 'program', 'question', 'work',
            'number', 'night', 'point', 'home', 'water', 'room', 'area', 'money', 'story',
            'fact', 'month', 'lot', 'right', 'study', 'book', 'eye', 'job', 'word',
            'business', 'issue', 'side', 'kind', 'head', 'house', 'service', 'friend',
            'power', 'hour', 'game', 'line', 'end', 'member', 'law', 'car', 'city',
            'community', 'name', 'team', 'minute', 'idea', 'body', 'information', 'back',
            'level', 'office', 'door', 'health', 'person', 'art', 'war', 'history',
            'party', 'result', 'change', 'morning', 'reason', 'research', 'education',
            # Domain-specific
            'programming', 'code', 'function', 'variable', 'data', 'algorithm',
            'software', 'hardware', 'computer', 'internet', 'technology',
            'artificial', 'intelligence', 'machine', 'learning', 'neural', 'network',
            'model', 'training', 'deep', 'math', 'mathematics', 'number',
            'equation', 'formula', 'geometry', 'algebra', 'statistics', 'probability',
            'philosophy', 'consciousness', 'ethics', 'moral', 'existence', 'meaning',
            'androm', 'system', 'brain', 'unit', 'rule', 'engine', 'memory',
            # Adjectives
            'good', 'better', 'best', 'bad', 'worse', 'great', 'new', 'old',
            'first', 'last', 'long', 'little', 'own', 'other', 'right', 'big', 'high',
            'different', 'small', 'large', 'next', 'early', 'important', 'few', 'public',
            'readable', 'maintainable', 'efficient', 'coherent', 'intelligent', 'smart',
            # Greetings
            'hello', 'hi', 'goodbye', 'bye', 'please', 'thank', 'thanks', 'sorry',
        ]
        
        for word in words:
            idx = len(self.token_to_id)
            if idx < self.vocab_size:
                self.token_to_id[word.lower()] = idx
                self.id_to_token[idx] = word
    
    def _init_word_vectors(self):
        """Initialize word vectors with semantic relationships."""
        np.random.seed(42)
        
        # Create base vectors for all vocabulary
        for word in self.token_to_id:
            if word not in ['<PAD>', '<UNK>', '<START>', '<END>']:
                self.word_vectors[word] = np.random.randn(self.embed_dim) * 0.1
        
        # Adjust vectors for semantically similar words
        semantic_groups = [
            ['programming', 'code', 'function', 'algorithm', 'software'],
            ['ai', 'artificial', 'intelligence', 'machine', 'learning', 'neural', 'network'],
            ['math', 'mathematics', 'number', 'equation', 'formula', 'algebra', 'statistics'],
            ['philosophy', 'ethics', 'moral', 'existence', 'meaning', 'consciousness'],
            ['technology', 'computer', 'hardware', 'internet', 'digital'],
            ['hello', 'hi', 'hey', 'greetings'],
            ['goodbye', 'bye', 'farewell'],
            ['androm', 'system', 'brain', 'self', 'improve', 'evolve'],
        ]
        
        for group in semantic_groups:
            # Make words in same group have similar vectors
            base = np.random.randn(self.embed_dim) * 0.1
            for word in group:
                if word in self.word_vectors:
                    self.word_vectors[word] = base + np.random.randn(self.embed_dim) * 0.02
    
    def tokenize(self, text: str) -> list[str]:
        """Smart tokenization."""
        # Lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return [w for w in words if len(w) > 1]
    
    def extract_keywords(self, text: str) -> list[str]:
        """Extract important keywords from text."""
        tokens = self.tokenize(text)
        # Filter stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                    'could', 'should', 'can', 'may', 'might', 'to', 'of', 'in',
                    'for', 'on', 'with', 'at', 'by', 'from', 'and', 'or', 'but'}
        return [t for t in tokens if t not in stopwords and len(t) > 2]
    
    def detect_topic(self, text: str) -> str:
        """Detect the topic of text."""
        tokens = set(self.tokenize(text))
        
        best_topic = "general"
        best_score = 0
        
        for topic, keywords in self.topic_keywords.items():
            overlap = len(tokens & keywords)
            if overlap > best_score:
                best_score = overlap
                best_topic = topic
        
        return best_topic
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to vector using word embeddings."""
        tokens = self.tokenize(text)
        if not tokens:
            return np.zeros(self.embed_dim)
        
        # Get word vectors
        vectors = []
        for token in tokens[:50]:
            if token in self.word_vectors:
                vectors.append(self.word_vectors[token])
            elif token in self.token_to_id:
                # Generate consistent vector for unknown words
                np.random.seed(hash(token) % 2**31)
                vectors.append(np.random.randn(self.embed_dim) * 0.1)
        
        if not vectors:
            return np.zeros(self.embed_dim)
        
        # Weighted average (longer words are often more important)
        weights = [len(t) for t in tokens[:50]]
        total_weight = sum(weights)
        
        if total_weight > 0:
            weighted_sum = sum(w * v for w, v in zip(weights, vectors))
            result = np.array(weighted_sum / total_weight, dtype=np.float64)
        else:
            result = np.array(np.mean(vectors, axis=0), dtype=np.float64)
        
        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        
        return result
    
    def word_overlap(self, text1: str, text2: str) -> float:
        """Calculate word overlap between two texts."""
        tokens1 = set(self.extract_keywords(text1))
        tokens2 = set(self.extract_keywords(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings."""
        vec1 = self.encode(text1)
        vec2 = self.encode(text2)
        
        dot = np.dot(vec1, vec2)
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        return dot / norm if norm > 0 else 0.0


class MemoryBank:
    """Smart memory bank with relevance-based retrieval."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.memories: list[Memory] = []
        self.encoder = SmartEncoder()
        self.generation_counter = 0
        
        # Topic index for fast lookup
        self.topic_index: dict[str, list[int]] = defaultdict(list)
    
    def store(self, text: str, memory_type: str = "response", topic: str | None = None):
        """Store text in memory with metadata."""
        if not text or len(text.strip()) < 3:
            return
        
        if topic is None:
            topic = self.encoder.detect_topic(text)
        
        key = self.encoder.encode(text)
        tokens = self.encoder.tokenize(text)
        memory = Memory(key=key, value=text, tokens=tokens, 
                       memory_type=memory_type, topic=topic,
                       created_at=self.generation_counter)
        
        idx = len(self.memories)
        self.memories.append(memory)
        self.topic_index[topic].append(idx)
        self.generation_counter += 1
        
        # Evict oldest if over capacity
        if len(self.memories) > self.max_size:
            self._evict_oldest()
    
    def _evict_oldest(self):
        """Evict oldest memory."""
        if self.memories:
            self.memories.pop(0)
            # Rebuild topic index
            self.topic_index.clear()
            for i, mem in enumerate(self.memories):
                self.topic_index[mem.topic].append(i)
    
    def retrieve(self, query: str, top_k: int = 5, 
                 memory_type: str | None = None,
                 same_topic_only: bool = False) -> list[tuple[Memory, float]]:
        """Smart retrieval with multiple relevance signals."""
        if not self.memories:
            return []
        
        query_topic = self.encoder.detect_topic(query)
        query_vec = self.encoder.encode(query)
        query_keywords = set(self.encoder.extract_keywords(query))
        
        candidates = self.memories
        if memory_type:
            candidates = [m for m in self.memories if m.memory_type == memory_type]
        
        # Score each memory
        scored = []
        for mem in candidates:
            # Skip exact matches
            if mem.value.lower() == query.lower():
                continue
            
            # Skip questions (we want answers)
            if '?' in mem.value and '?' in query:
                continue
            
            # Calculate multiple relevance signals
            semantic_score = mem.similarity(query_vec)
            word_overlap = self.encoder.word_overlap(query, mem.value)
            topic_match = 1.0 if mem.topic == query_topic else 0.2
            
            # Keyword overlap bonus
            mem_keywords = set(self.encoder.extract_keywords(mem.value))
            keyword_overlap = len(query_keywords & mem_keywords) / max(len(query_keywords), 1)
            
            # Combined score (weighted)
            combined = (semantic_score * 0.3 + word_overlap * 0.3 + topic_match * 0.2 + keyword_overlap * 0.2)
            
            # Boost for frequently accessed memories
            access_boost = min(mem.access_count * 0.005, 0.05)
            combined += access_boost
            
            # Penalty for very short responses to complex questions
            if len(query.split()) > 3 and len(mem.value.split()) < 5:
                combined *= 0.7
            
            scored.append((mem, combined))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Update access counts
        for mem, score in scored[:top_k]:
            mem.access_count += 1
        
        return scored[:top_k]
    
    def store_many(self, texts: list[str], memory_type: str = "response"):
        """Store multiple texts."""
        for text in texts:
            self.store(text, memory_type)
    
    def size(self) -> int:
        return len(self.memories)


class MemNet:
    """
    Advanced Memory-Augmented Neural Network.
    
    Features:
    - Smart semantic encoding
    - Topic-aware retrieval
    - Multi-signal relevance scoring
    - Contextual response generation
    """
    
    def __init__(self, memory_size: int = 10000):
        self.encoder = SmartEncoder()
        self.memory_bank = MemoryBank(max_size=memory_size)
        self.conversation_context: list[str] = []
        
        # Response templates
        self.templates = self._init_templates()
    
    def _init_templates(self) -> dict[str, list[str]]:
        """Initialize response templates."""
        return {
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! What would you like to know?",
                "Greetings! I'm ready to assist you.",
                "Hey! What's on your mind?",
            ],
            "farewell": [
                "Goodbye! Have a great day!",
                "See you later! Take care!",
                "Bye! Feel free to come back anytime.",
            ],
            "unknown": [
                "That's an interesting question. Let me think about it.",
                "I'm not sure about that, but I'm always learning.",
                "Could you tell me more about what you mean?",
            ],
        }
    
    def train(self, texts: list[str]):
        """Train on texts."""
        self.memory_bank.store_many(texts, memory_type="response")
    
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate a smart, relevant response."""
        # Update context
        self.conversation_context.append(prompt)
        if len(self.conversation_context) > 5:
            self.conversation_context.pop(0)
        
        # Detect intent
        intent = self._detect_intent(prompt)
        
        # Get relevant memories
        retrieved = self.memory_bank.retrieve(prompt, top_k=10)
        
        # Filter for relevance and exclude questions
        relevant = [(mem, score) for mem, score in retrieved 
                   if score > 0.2 and '?' not in mem.value]
        
        if relevant:
            # Pick best response that's not just echoing the question
            for mem, score in relevant:
                # Skip if response is too similar to question (just rephrasing)
                prompt_words = set(prompt.lower().split())
                response_words = set(mem.value.lower().split())
                overlap = len(prompt_words & response_words) / max(len(prompt_words), 1)
                
                if overlap < 0.6:
                    return mem.value
            
            # If all responses are too similar, use best one
            return relevant[0][0].value
        
        # Fallback to templates
        return self._template_response(intent)
    
    def _detect_intent(self, text: str) -> str:
        """Detect user intent."""
        text_lower = text.lower()
        
        if any(g in text_lower for g in ['hello', 'hi ', 'hey', 'greetings']):
            return "greeting"
        elif any(f in text_lower for f in ['bye', 'goodbye', 'see you']):
            return "farewell"
        elif '?' in text:
            return "question"
        else:
            return "statement"
    
    def _smart_response(self, memories: list[tuple[Memory, float]], 
                       context: str, intent: str) -> str:
        """Generate a smart response from multiple memories."""
        context_lower = context.lower()
        context_keywords = set(self.encoder.extract_keywords(context))
        
        # Check for specific topics that need direct answers
        topic_handlers = {
            'androm': lambda: "I am ANDROM, an Adaptive Network of Deterministic Rule-based Operations and Mathematics. I use thousands of computational units to process information.",
            'recursion': lambda: "Recursion is when a function calls itself to solve smaller instances of the same problem. It's useful for problems that can be broken into similar subproblems.",
            'consciousness': lambda: "Consciousness is the awareness of ourselves and our surroundings. Its nature is one of philosophy's deepest questions, debated by thinkers for centuries.",
        }
        
        for keyword, handler in topic_handlers.items():
            if keyword in context_lower:
                return handler()
        
        # Find memories that directly answer the question
        answers = []
        for mem, score in memories:
            # Skip questions
            if '?' in mem.value:
                continue
            
            mem_keywords = set(self.encoder.extract_keywords(mem.value))
            overlap = len(context_keywords & mem_keywords)
            
            # Check if this looks like an answer (not a question, has content)
            is_answer = len(mem.value) > 20
            
            if is_answer and overlap > 0:
                answers.append((mem, score, overlap))
        
        if answers:
            # Sort by overlap and score
            answers.sort(key=lambda x: (x[2], x[1]), reverse=True)
            return answers[0][0].value
        
        # If no direct answers, find best non-question memory
        non_questions = [(mem, score) for mem, score in memories if '?' not in mem.value]
        if non_questions:
            return non_questions[0][0].value
        
        # Fallback
        return memories[0][0].value if memories else self._template_response(intent)
    
    def _template_response(self, intent: str) -> str:
        """Get template response for intent."""
        if intent in self.templates:
            return random.choice(self.templates[intent])
        return random.choice(self.templates["unknown"])
    
    def get_stats(self) -> dict:
        """Get model statistics."""
        return {
            "memory_size": self.memory_bank.size(),
            "vocab_size": len(self.encoder.token_to_id),
            "topics": list(self.encoder.topic_keywords.keys()),
        }
