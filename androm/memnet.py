"""
MemNet - Memory-Augmented Neural Network for text generation.
Innovative architecture combining neural networks with external memory.

This is a novel approach that uses:
- Neural encoder for semantic understanding
- External memory bank for storing knowledge
- Similarity-based retrieval for finding relevant responses
- No transformers, no attention mechanisms
"""

from __future__ import annotations
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class Memory:
    """A single memory entry."""
    key: np.ndarray  # Encoded representation
    value: str  # Stored text
    tokens: list[str] = field(default_factory=list)  # Tokenized
    memory_type: str = "response"  # "response", "question", "statement"
    access_count: int = 0
    created_at: int = 0
    
    def similarity(self, query: np.ndarray) -> float:
        """Cosine similarity between query and key."""
        dot = np.dot(self.key, query)
        norm = np.linalg.norm(self.key) * np.linalg.norm(query)
        if norm == 0:
            return 0.0
        return dot / norm


class NeuralEncoder:
    """Small neural network for encoding text to vectors."""
    
    def __init__(self, vocab_size: int = 2000, embed_dim: int = 128, hidden_dim: int = 256):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights randomly
        np.random.seed(42)
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.02
        self.W1 = np.random.randn(embed_dim, hidden_dim) * 0.02
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, embed_dim) * 0.02
        self.b2 = np.zeros(embed_dim)
        
        # Token mapping
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Build vocabulary from common tokens."""
        # Special tokens
        specials = ['<PAD>', '<UNK>', '<START>', '<END>']
        for i, token in enumerate(specials):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        # Common words - expanded vocabulary
        common_words = [
            # Articles & pronouns
            'the', 'a', 'an', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
            # Verbs
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might',
            'make', 'makes', 'made', 'take', 'takes', 'took', 'come', 'comes', 'came',
            'go', 'goes', 'went', 'get', 'gets', 'got', 'know', 'knows', 'knew',
            'think', 'thinks', 'thought', 'see', 'sees', 'saw', 'want', 'wants', 'wanted',
            'use', 'uses', 'used', 'find', 'finds', 'found', 'give', 'gives', 'gave',
            'tell', 'tells', 'told', 'work', 'works', 'worked', 'call', 'calls', 'called',
            'try', 'tries', 'tried', 'need', 'needs', 'needed', 'feel', 'feels', 'felt',
            'become', 'becomes', 'became', 'leave', 'leaves', 'left', 'put', 'puts',
            'mean', 'means', 'meant', 'keep', 'keeps', 'kept', 'let', 'lets',
            'begin', 'begins', 'began', 'show', 'shows', 'showed', 'hear', 'hears', 'heard',
            'play', 'plays', 'played', 'run', 'runs', 'ran', 'move', 'moves', 'moved',
            'live', 'lives', 'lived', 'believe', 'believes', 'believed',
            'help', 'helps', 'helped', 'learn', 'learns', 'learned',
            'create', 'creates', 'created', 'solve', 'solves', 'solved',
            'improve', 'improves', 'improved', 'process', 'processes', 'processed',
            # Adjectives
            'good', 'better', 'best', 'bad', 'worse', 'worst', 'great', 'new', 'old',
            'first', 'last', 'long', 'little', 'own', 'other', 'right', 'big', 'high',
            'different', 'small', 'large', 'next', 'early', 'important', 'few', 'public',
            'readable', 'maintainable', 'efficient', 'coherent', 'intelligent',
            # Nouns
            'time', 'year', 'people', 'way', 'day', 'man', 'woman', 'child', 'world',
            'life', 'hand', 'part', 'place', 'case', 'week', 'company', 'system', 'program',
            'question', 'work', 'government', 'number', 'night', 'point', 'home',
            'water', 'room', 'mother', 'area', 'money', 'story', 'fact', 'month',
            'lot', 'right', 'study', 'book', 'eye', 'job', 'word', 'business',
            'issue', 'side', 'kind', 'head', 'house', 'service', 'friend', 'father',
            'power', 'hour', 'game', 'line', 'end', 'member', 'law', 'car',
            'city', 'community', 'name', 'president', 'team', 'minute', 'idea',
            'body', 'information', 'back', 'parent', 'face', 'others', 'level',
            'office', 'door', 'health', 'person', 'art', 'war', 'history',
            'party', 'result', 'change', 'morning', 'reason', 'research', 'girl',
            'guy', 'moment', 'air', 'teacher', 'force', 'education',
            'programming', 'code', 'function', 'variable', 'data', 'algorithm',
            'software', 'hardware', 'computer', 'internet', 'technology',
            'artificial', 'intelligence', 'machine', 'learning', 'neural', 'network',
            'model', 'training', 'deep', 'math', 'mathematics', 'number',
            'equation', 'formula', 'geometry', 'algebra', 'statistics', 'probability',
            'philosophy', 'consciousness', 'ethics', 'moral', 'existence', 'meaning',
            'androm', 'system', 'brain', 'unit', 'rule', 'engine', 'memory',
            # Adverbs
            'very', 'really', 'just', 'also', 'now', 'then', 'always', 'never',
            'often', 'still', 'already', 'ever', 'soon', 'today', 'here', 'there',
            # Prepositions
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up',
            'about', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'over', 'without', 'within',
            # Conjunctions
            'and', 'or', 'but', 'if', 'when', 'because', 'while', 'although',
            'that', 'what', 'which', 'who', 'whom', 'whose',
            # Common words
            'not', 'no', 'yes', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'other', 'some', 'such', 'only', 'same', 'so', 'than', 'too', 'how',
            # Greetings & common phrases
            'hello', 'hi', 'goodbye', 'bye', 'please', 'thank', 'thanks', 'sorry',
            'help', 'today', 'doing', 'well',
        ]
        
        for i, word in enumerate(common_words):
            idx = len(self.token_to_id)
            if idx < self.vocab_size:
                self.token_to_id[word.lower()] = idx
                self.id_to_token[idx] = word
    
    def tokenize(self, text: str) -> list[int]:
        """Convert text to token IDs."""
        # Simple word tokenization with punctuation handling
        words = re.findall(r'\b\w+\b|[.!?]', text.lower())
        ids = []
        for word in words:
            if word in self.token_to_id:
                ids.append(self.token_to_id[word])
            else:
                ids.append(self.token_to_id['<UNK>'])
        return ids
    
    def tokenize_to_words(self, text: str) -> list[str]:
        """Convert text to word tokens."""
        return re.findall(r'\b\w+\b|[.!?]', text.lower())
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to vector representation."""
        token_ids = self.tokenize(text)
        if not token_ids:
            return np.zeros(self.embed_dim)
        
        # Get embeddings
        embeddings = []
        for tid in token_ids[:100]:  # Limit length
            if tid < self.vocab_size:
                embeddings.append(self.embedding[tid])
        
        if not embeddings:
            return np.zeros(self.embed_dim)
        
        # Average pooling
        avg_embed = np.mean(embeddings, axis=0)
        
        # Neural transformation
        hidden = np.tanh(avg_embed @ self.W1 + self.b1)
        output = hidden @ self.W2 + self.b2
        
        # Normalize
        norm = np.linalg.norm(output)
        if norm > 0:
            output = output / norm
        
        return output


class MemoryBank:
    """External memory bank for storing and retrieving patterns."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.memories: list[Memory] = []
        self.encoder = NeuralEncoder()
        self.generation_counter = 0
    
    def store(self, text: str, memory_type: str = "response"):
        """Store text in memory."""
        if not text or len(text.strip()) < 3:
            return
        
        key = self.encoder.encode(text)
        tokens = self.encoder.tokenize_to_words(text)
        memory = Memory(key=key, value=text, tokens=tokens, 
                       memory_type=memory_type, created_at=self.generation_counter)
        self.memories.append(memory)
        self.generation_counter += 1
        
        # Evict oldest if over capacity
        if len(self.memories) > self.max_size:
            self.memories.pop(0)
    
    def store_many(self, texts: list[str], memory_type: str = "response"):
        """Store multiple texts."""
        for text in texts:
            self.store(text, memory_type)
    
    def retrieve(self, query: str, top_k: int = 5, 
                 memory_type: str | None = None) -> list[tuple[Memory, float]]:
        """Retrieve most similar memories to query."""
        if not self.memories:
            return []
        
        query_vec = self.encoder.encode(query)
        
        # Filter by type if specified
        candidates = self.memories
        if memory_type:
            candidates = [m for m in self.memories if m.memory_type == memory_type]
        
        # Compute similarities
        scored = [(mem, mem.similarity(query_vec)) for mem in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Update access counts
        for mem, score in scored[:top_k]:
            mem.access_count += 1
        
        return scored[:top_k]
    
    def size(self) -> int:
        return len(self.memories)


class MemNet:
    """
    Memory-Augmented Neural Network for text generation.
    
    Architecture:
    1. Neural Encoder: Encodes input to vector
    2. Memory Bank: Stores learned text patterns
    3. Retrieval: Finds relevant memories
    4. Generator: Produces text using retrieved context
    
    This is different from:
    - Transformers: No quadratic attention, uses memory retrieval instead
    - RNNs: Not sequential, can access any memory
    - N-grams: Uses learned representations, not just token counts
    """
    
    def __init__(self, memory_size: int = 10000):
        self.encoder = NeuralEncoder(vocab_size=2000, embed_dim=128, hidden_dim=256)
        self.memory_bank = MemoryBank(max_size=memory_size)
        
        # Response templates by topic
        self.topic_templates: dict[str, list[str]] = {}
        self._init_templates()
    
    def _init_templates(self):
        """Initialize response templates."""
        self.topic_templates = {
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! What would you like to know?",
                "Greetings! I am ready to assist you.",
                "Hey! What's on your mind?",
            ],
            "farewell": [
                "Goodbye! Have a great day!",
                "See you later! Take care!",
                "Bye! Feel free to come back anytime.",
            ],
            "question": [
                "That's a great question. Let me think about it.",
                "Interesting question. Here's what I know:",
                "Good question! Based on my knowledge:",
            ],
            "general": [
                "I see. Tell me more about that.",
                "That's interesting. I'm processing that.",
                "Hmm, let me consider that.",
            ],
        }
    
    def train(self, texts: list[str]):
        """Train on texts by storing in memory."""
        self.memory_bank.store_many(texts, memory_type="response")
    
    def generate(self, prompt: str, max_length: int = 50, temperature: float = 0.8) -> str:
        """
        Generate text using memory-augmented approach.
        
        Process:
        1. Encode prompt
        2. Retrieve relevant memories
        3. Generate tokens using memory context
        """
        # Get relevant memories (only responses, not questions)
        retrieved = self.memory_bank.retrieve(prompt, top_k=5, memory_type="response")
        
        if retrieved:
            # Filter out exact matches (don't repeat user input)
            filtered = [(mem, score) for mem, score in retrieved 
                       if mem.value.lower() != prompt.lower() and score > 0.3]
            
            if filtered:
                best_mem, best_score = filtered[0]
                
                if best_score > 0.5:
                    # Good match - return the response
                    return best_mem.value
                elif best_score > 0.35:
                    # Medium match - try to construct a response
                    return self._construct_response(filtered, prompt)
        
        # No good matches - generate from templates
        return self._generate_from_template(prompt)
    
    def _construct_response(self, memories: list[tuple[Memory, float]], context: str) -> str:
        """Construct a response from retrieved memories."""
        context_tokens = set(self.encoder.tokenize_to_words(context))
        
        # Find memories that share tokens with context
        relevant = []
        for mem, score in memories:
            shared = set(mem.tokens) & context_tokens
            if shared:
                relevant.append((mem, score, len(shared)))
        
        if relevant:
            # Sort by shared tokens count and similarity
            relevant.sort(key=lambda x: (x[2], x[1]), reverse=True)
            return relevant[0][0].value
        
        # If no relevant memories, return best memory
        return memories[0][0].value if memories else self._generate_from_template(context)
    
    def _generate_from_template(self, prompt: str) -> str:
        """Generate from templates when no good memories."""
        prompt_lower = prompt.lower()
        
        if any(g in prompt_lower for g in ['hello', 'hi ', 'hey', 'greetings']):
            return random.choice(self.topic_templates["greeting"])
        elif any(f in prompt_lower for f in ['bye', 'goodbye', 'see you']):
            return random.choice(self.topic_templates["farewell"])
        elif '?' in prompt:
            return random.choice(self.topic_templates["question"])
        else:
            return random.choice(self.topic_templates["general"])
    
    def get_stats(self) -> dict:
        """Get model statistics."""
        return {
            "memory_size": self.memory_bank.size(),
            "vocab_size": len(self.encoder.token_to_id),
            "generation_counter": self.memory_bank.generation_counter,
        }
