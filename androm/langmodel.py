"""
LangModel - Word-by-word language model for ANDROM.
Predicts next word given context, generates text token by token.
"""

from __future__ import annotations
import math
import random
import re
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class Vocab:
    """Vocabulary for tokenization."""
    word_to_id: dict[str, int] = field(default_factory=dict)
    id_to_word: dict[int, str] = field(default_factory=dict)
    word_counts: Counter = field(default_factory=Counter)
    
    PAD = "<PAD>"
    UNK = "<UNK>"
    START = "<START>"
    END = "<END>"
    
    def __post_init__(self):
        # Add special tokens
        for token in [self.PAD, self.UNK, self.START, self.END]:
            if token not in self.word_to_id:
                self.add_token(token)
    
    def add_token(self, token: str) -> int:
        """Add a token to vocabulary."""
        if token not in self.word_to_id:
            idx = len(self.word_to_id)
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
        self.word_counts[token] += 1
        return self.word_to_id[token]
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        unk_id = self.word_to_id[self.UNK]
        return [self.word_to_id.get(t, unk_id) for t in tokens]
    
    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        tokens = []
        for idx in ids:
            if idx in self.id_to_word:
                token = self.id_to_word[idx]
                if token not in [self.PAD, self.UNK, self.START, self.END]:
                    tokens.append(token)
        return self._detokenize(tokens)
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        # Lowercase and split on whitespace/punctuation
        text = text.lower()
        # Split but keep punctuation as separate tokens
        tokens = re.findall(r'\b\w+\b|[.!?,:;]', text)
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
    
    def size(self) -> int:
        return len(self.word_to_id)
    
    def build_from_texts(self, texts: list[str], min_count: int = 1):
        """Build vocabulary from texts."""
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                self.add_token(token)
        
        # Remove rare words if needed
        if min_count > 1:
            rare = [w for w, c in self.word_counts.items() if c < min_count]
            # Keep special tokens
            special = {self.PAD, self.UNK, self.START, self.END}
            for w in rare:
                if w not in special and w in self.word_to_id:
                    del self.word_to_id[w]
                    del self.word_counts[w]
            # Rebuild id_to_word
            self.id_to_word = {v: k for k, v in self.word_to_id.items()}


class SmallLM:
    """
    Small neural language model.
    
    Uses a feedforward neural network with context window to predict next word.
    Trained on text data to learn word patterns.
    
    Architecture:
    - Input: Context window of previous words (as embeddings)
    - Hidden: Neural network layers
    - Output: Probability distribution over vocabulary
    
    This is a simplified version of how large language models work.
    """
    
    def __init__(self, vocab_size: int = 5000, embed_dim: int = 64, 
                 hidden_dim: int = 128, context_size: int = 5):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size
        
        # Initialize weights
        np.random.seed(42)
        
        # Embedding matrix: vocab_size x embed_dim
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.01
        
        # Hidden layer: (context_size * embed_dim) x hidden_dim
        input_dim = context_size * embed_dim
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        
        # Output layer: hidden_dim x vocab_size
        self.W2 = np.random.randn(hidden_dim, vocab_size) * 0.01
        self.b2 = np.zeros(vocab_size)
        
        # Vocabulary
        self.vocab = Vocab()
        
        # Training stats
        self.train_loss: list[float] = []
        self.is_trained = False
    
    def _forward(self, context_ids: list[int]) -> np.ndarray:
        """Forward pass: predict next word probabilities."""
        # Get embeddings for context words
        embeddings = []
        for word_id in context_ids[-self.context_size:]:
            if word_id < self.vocab_size:
                embeddings.append(self.embedding[word_id])
            else:
                embeddings.append(np.zeros(self.embed_dim))
        
        # Pad if needed
        while len(embeddings) < self.context_size:
            embeddings.insert(0, np.zeros(self.embed_dim))
        
        # Concatenate context embeddings
        x = np.concatenate(embeddings)
        
        # Hidden layer with ReLU
        h = np.maximum(0, x @ self.W1 + self.b1)
        
        # Output layer (logits)
        logits = h @ self.W2 + self.b2
        
        # Softmax to get probabilities
        probs = self._softmax(logits)
        
        return probs
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def train(self, texts: list[str], epochs: int = 10, lr: float = 0.01,
              batch_size: int = 32):
        """
        Train the language model on texts.
        
        Args:
            texts: List of training texts
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size for training
        """
        # Build vocabulary
        self.vocab.build_from_texts(texts)
        
        # Prepare training data
        sequences = []
        for text in texts:
            ids = self.vocab.encode(text)
            if len(ids) > self.context_size:
                sequences.append(ids)
        
        if not sequences:
            print("No training data!")
            return
        
        print(f"Training on {len(sequences)} sequences, vocab size: {self.vocab.size()}")
        
        # Training loop
        for epoch in range(epochs):
            total_loss: float = 0.0
            num_batches = 0
            
            # Shuffle sequences
            random.shuffle(sequences)
            
            for seq in sequences:
                # Create training pairs (context, target)
                for i in range(len(seq) - 1):
                    # Context is previous words
                    start = max(0, i - self.context_size + 1)
                    context = seq[start:i+1]
                    target = seq[i+1]
                    
                    # Forward pass
                    probs = self._forward(context)
                    
                    # Compute loss (cross-entropy)
                    loss = -np.log(probs[target] + 1e-10)
                    total_loss += loss
                    
                    # Backward pass (simplified gradient descent)
                    self._backward(context, target, probs, lr)
                    
                    num_batches += 1
            
            avg_loss = float(total_loss / max(num_batches, 1))
            self.train_loss.append(avg_loss)
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        print("Training complete!")
    
    def _backward(self, context: list[int], target: int, probs: np.ndarray, lr: float):
        """Backward pass: update weights."""
        # Get embeddings
        embeddings = []
        for word_id in context[-self.context_size:]:
            if word_id < self.vocab_size:
                embeddings.append(self.embedding[word_id])
            else:
                embeddings.append(np.zeros(self.embed_dim))
        
        while len(embeddings) < self.context_size:
            embeddings.insert(0, np.zeros(self.embed_dim))
        
        x = np.concatenate(embeddings)
        
        # Hidden layer
        h = np.maximum(0, x @ self.W1 + self.b1)
        
        # Output gradient
        d_logits = probs.copy()
        d_logits[target] -= 1  # Gradient of cross-entropy
        
        # Update output layer
        self.W2 -= lr * np.outer(h, d_logits)
        self.b2 -= lr * d_logits
        
        # Hidden gradient
        d_h = d_logits @ self.W2.T
        d_h[h <= 0] = 0  # ReLU gradient
        
        # Update hidden layer
        self.W1 -= lr * np.outer(x, d_h)
        self.b1 -= lr * d_h
    
    def generate(self, prompt: str = "", max_length: int = 50, 
                 temperature: float = 0.8) -> str:
        """
        Generate text word by word.
        
        Args:
            prompt: Starting text
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated text
        """
        if not self.is_trained:
            return "Model not trained yet."
        
        # Encode prompt
        if prompt:
            ids = self.vocab.encode(prompt)
        else:
            # Start with START token
            ids = [self.vocab.word_to_id.get(self.vocab.START, 0)]
        
        # Generate tokens one by one
        for _ in range(max_length):
            # Get context
            context = ids[-self.context_size:]
            
            # Predict next word probabilities
            probs = self._forward(context)
            
            # Apply temperature
            if temperature != 1.0:
                log_probs = np.log(probs + 1e-10)
                log_probs = log_probs / temperature
                probs = self._softmax(log_probs)
            
            # Sample next token
            next_id = self._sample(probs)
            
            # Stop if END token
            if next_id == self.vocab.word_to_id.get(self.vocab.END, -1):
                break
            
            ids.append(next_id)
            
            # Stop if we generated a sentence ending
            word = self.vocab.id_to_word.get(next_id, "")
            if word in '.!?' and len(ids) > 5:
                break
        
        # Decode to text
        text = self.vocab.decode(ids)
        
        # Clean up
        if text:
            text = text[0].upper() + text[1:]
        
        return text
    
    def _sample(self, probs: np.ndarray) -> int:
        """Sample from probability distribution."""
        # Avoid numerical issues
        probs = np.maximum(probs, 0)
        probs = probs / probs.sum()
        
        return np.random.choice(len(probs), p=probs)
    
    def predict_next(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Predict top-k next words given text."""
        ids = self.vocab.encode(text)
        context = ids[-self.context_size:]
        
        probs = self._forward(context)
        
        # Get top-k predictions
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            word = self.vocab.id_to_word.get(idx, self.vocab.UNK)
            prob = probs[idx]
            predictions.append((word, float(prob)))
        
        return predictions
    
    def get_stats(self) -> dict:
        """Get model statistics."""
        return {
            "vocab_size": self.vocab.size(),
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "context_size": self.context_size,
            "is_trained": self.is_trained,
            "train_loss": self.train_loss[-1] if self.train_loss else None,
        }
