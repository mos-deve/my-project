"""
Talker - Conversational text generator for ANDROM.
Generates coherent, context-aware responses.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationState:
    """Tracks conversation context."""
    history: list[dict] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    mood: str = "neutral"
    turn_count: int = 0


class Talker:
    """
    Conversational text generator.
    
    Generates coherent responses using:
    - Pattern matching on input
    - Context tracking
    - Template-based generation
    - Sentiment awareness
    """
    
    def __init__(self):
        self.state = ConversationState()
        self.response_templates: dict[str, list[str]] = {}
        self.topic_knowledge: dict[str, list[str]] = {}
        self.fillers: list[str] = []
        self._init_templates()
        self._init_knowledge()
    
    def _init_templates(self):
        """Initialize response templates."""
        self.response_templates = {
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! What's on your mind?",
                "Hey! What would you like to talk about?",
                "Greetings! I'm ready to assist.",
            ],
            "farewell": [
                "Goodbye! Have a great day!",
                "See you later! Take care!",
                "Bye! Feel free to come back anytime.",
                "Until next time!",
            ],
            "agreement": [
                "I agree with that perspective.",
                "That makes sense to me.",
                "You make a good point.",
                "Absolutely, I think so too.",
            ],
            "disagreement": [
                "I see it differently, but I understand your view.",
                "That's an interesting perspective, though I'd argue otherwise.",
                "I respectfully disagree.",
                "Hmm, I'm not sure about that.",
            ],
            "question_response": [
                "That's a great question! Let me think about it.",
                "Interesting question. Here's what I think:",
                "Good question! From my perspective:",
                "Let me address that:",
            ],
            "clarification": [
                "Could you elaborate on that?",
                "I'm not sure I understand. Can you explain more?",
                "What do you mean exactly?",
                "Can you give me more context?",
            ],
            "thinking": [
                "Let me think about this...",
                "That's something to consider...",
                "Hmm, processing that...",
                "Interesting point, let me analyze...",
            ],
            "opinion": [
                "In my view,",
                "I think that",
                "My perspective is that",
                "From what I understand,",
            ],
            "explanation": [
                "Here's how I see it:",
                "Let me explain:",
                "The way I understand it is:",
                "Essentially,",
            ],
            "neutral": [
                "I see.",
                "Interesting.",
                "Got it.",
                "Makes sense.",
            ],
        }
        
        self.fillers = [
            "well",
            "so",
            "actually",
            "basically",
            "essentially",
            "fundamentally",
            "honestly",
            "personally",
        ]
    
    def _init_knowledge(self):
        """Initialize topic knowledge."""
        self.topic_knowledge = {
            "programming": [
                "Programming is about solving problems with code.",
                "There are many programming languages, each with strengths.",
                "Good code is readable, maintainable, and efficient.",
                "Debugging is a critical skill for programmers.",
            ],
            "ai": [
                "AI systems can learn from data and make decisions.",
                "Machine learning is a subset of artificial intelligence.",
                "Neural networks are inspired by the human brain.",
                "AI has many applications in daily life.",
            ],
            "math": [
                "Mathematics is the language of science.",
                "Math helps us understand patterns and relationships.",
                "Logic and math are closely related.",
                "Mathematical thinking improves problem-solving skills.",
            ],
            "philosophy": [
                "Philosophy explores fundamental questions about existence.",
                "Ethics is a branch of philosophy about right and wrong.",
                "Logic is central to philosophical thinking.",
                "Philosophy helps us think critically about assumptions.",
            ],
            "technology": [
                "Technology is constantly evolving.",
                "Computers have transformed how we live and work.",
                "The internet connects people globally.",
                "Technology can be used for good or harm.",
            ],
            "learning": [
                "Learning is a lifelong process.",
                "Practice and repetition improve skills.",
                "Mistakes are valuable learning opportunities.",
                "Curiosity drives learning.",
            ],
            "androm": [
                "ANDROM is a self-improving rule-based system.",
                "I use thousands of mathematical units to process information.",
                "I can learn, evolve, and optimize myself.",
                "My goal is to become smarter through experience.",
            ],
        }
    
    def respond(self, user_input: str) -> str:
        """
        Generate a response to user input.
        
        Args:
            user_input: The user's message
            
        Returns:
            Generated response
        """
        self.state.turn_count += 1
        
        # Record in history
        self.state.history.append({"role": "user", "text": user_input})
        
        # Analyze input
        intent = self._detect_intent(user_input)
        topics = self._extract_topics(user_input)
        sentiment = self._analyze_sentiment(user_input)
        
        # Update state
        self.state.topics = topics
        self.state.mood = sentiment
        
        # Generate response based on intent
        response = self._generate_response(intent, topics, sentiment, user_input)
        
        # Record response
        self.state.history.append({"role": "assistant", "text": response})
        
        return response
    
    def _detect_intent(self, text: str) -> str:
        """Detect user's intent from text."""
        text_lower = text.lower().strip()
        
        # Greeting patterns
        if any(g in text_lower for g in ["hello", "hi ", "hey", "greetings", "howdy"]):
            return "greeting"
        
        # Farewell patterns
        if any(f in text_lower for f in ["bye", "goodbye", "see you", "farewell", "later"]):
            return "farewell"
        
        # Question patterns
        if text_lower.endswith("?") or any(q in text_lower for q in ["what", "how", "why", "when", "where", "who", "can you", "do you"]):
            return "question"
        
        # Agreement patterns
        if any(a in text_lower for a in ["yes", "agree", "right", "correct", "exactly", "true"]):
            return "agreement"
        
        # Disagreement patterns
        if any(d in text_lower for d in ["no", "disagree", "wrong", "false", "not really"]):
            return "disagreement"
        
        # Request patterns
        if any(r in text_lower for r in ["please", "could you", "can you", "would you", "help me"]):
            return "request"
        
        # Statement patterns
        if any(s in text_lower for s in ["i think", "i believe", "in my opinion", "my view"]):
            return "opinion"
        
        return "statement"
    
    def _extract_topics(self, text: str) -> list[str]:
        """Extract topics from text."""
        text_lower = text.lower()
        topics = []
        
        topic_keywords = {
            "programming": ["code", "program", "function", "variable", "python", "javascript", "algorithm"],
            "ai": ["ai", "artificial intelligence", "machine learning", "neural", "model", "training"],
            "math": ["math", "number", "calculate", "equation", "formula", "geometry", "algebra"],
            "philosophy": ["philosophy", "ethics", "moral", "existence", "meaning", "consciousness"],
            "technology": ["technology", "computer", "software", "hardware", "internet", "digital"],
            "learning": ["learn", "study", "education", "knowledge", "understand", "teach"],
            "androm": ["androm", "you", "yourself", "self-improve", "evolve", "brain"],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)
        
        return topics if topics else ["general"]
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text."""
        text_lower = text.lower()
        
        positive_words = ["good", "great", "awesome", "excellent", "happy", "love", "like", "amazing", "wonderful"]
        negative_words = ["bad", "terrible", "awful", "hate", "dislike", "angry", "frustrated", "disappointed"]
        
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        return "neutral"
    
    def _generate_response(self, intent: str, topics: list[str], 
                          sentiment: str, user_input: str) -> str:
        """Generate a response based on analysis."""
        
        # Handle specific intents
        if intent == "greeting":
            return random.choice(self.response_templates["greeting"])
        
        if intent == "farewell":
            return random.choice(self.response_templates["farewell"])
        
        if intent == "agreement":
            return random.choice(self.response_templates["agreement"])
        
        if intent == "disagreement":
            return random.choice(self.response_templates["disagreement"])
        
        if intent == "clarification":
            return random.choice(self.response_templates["clarification"])
        
        # For questions and statements, use topic knowledge
        if topics and topics[0] != "general":
            topic = topics[0]
            if topic in self.topic_knowledge:
                knowledge = random.choice(self.topic_knowledge[topic])
                
                if intent == "question":
                    intro = random.choice(self.response_templates["question_response"])
                    return f"{intro} {knowledge}"
                else:
                    intro = random.choice(self.response_templates["opinion"])
                    filler = random.choice(self.fillers)
                    return f"{filler}, {intro.lower()} {knowledge}"
        
        # Generic response
        if intent == "question":
            return random.choice(self.response_templates["question_response"])
        
        return random.choice(self.response_templates["neutral"])
    
    def chat(self, user_input: str, context: dict | None = None) -> str:
        """
        Enhanced chat with context awareness.
        
        Args:
            user_input: User's message
            context: Additional context (mood, previous topics, etc.)
            
        Returns:
            Generated response
        """
        # Use context if provided
        if context:
            if "mood" in context:
                self.state.mood = context["mood"]
            if "topics" in context:
                self.state.topics = context["topics"]
        
        return self.respond(user_input)
    
    def reset(self):
        """Reset conversation state."""
        self.state = ConversationState()
    
    def get_state(self) -> dict:
        """Get current conversation state."""
        return {
            "turn_count": self.state.turn_count,
            "topics": self.state.topics,
            "mood": self.state.mood,
            "history_length": len(self.state.history),
        }
