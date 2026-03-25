"""
Learner - Reinforcement learning for ANDROM.
Rewards good solutions, punishes bad ones, adjusts weights.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Any, Callable
from androm.network import Network
from androm.unit import Unit


@dataclass
class Experience:
    """A single learning experience."""
    inputs: list[float]
    outputs: list[float]
    reward: float
    context: dict = field(default_factory=dict)


class Learner:
    """
    Reinforcement learning system for ANDROM.
    
    Learns by:
    - Rewarding outputs that lead to good solutions
    - Punishing outputs that lead to bad solutions
    - Adjusting network weights based on rewards
    - Evolving rules that produce good outcomes
    """
    
    def __init__(self, network: Network, learning_rate: float = 0.01,
                 discount_factor: float = 0.95):
        self.network = network
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.experiences: list[Experience] = []
        self.reward_history: list[float] = []
        self.total_reward: float = 0.0
    
    def record(self, inputs: list[float], outputs: list[float], 
               reward: float, context: dict | None = None):
        """Record an experience with reward signal."""
        exp = Experience(
            inputs=inputs,
            outputs=outputs,
            reward=reward,
            context=context or {},
        )
        self.experiences.append(exp)
        self.reward_history.append(reward)
        self.total_reward += reward
    
    def reward(self, amount: float = 1.0):
        """Give positive reward for last action."""
        if self.experiences:
            self.experiences[-1].reward += amount
            self.total_reward += amount
    
    def punish(self, amount: float = 1.0):
        """Give negative reward for last action."""
        if self.experiences:
            self.experiences[-1].reward -= amount
            self.total_reward -= amount
    
    def learn_from_experiences(self, batch_size: int = 32):
        """
        Learn from recent experiences using simplified policy gradient.
        
        Adjusts weights proportionally to reward received.
        """
        if len(self.experiences) < batch_size:
            return
        
        # Get recent batch
        batch = self.experiences[-batch_size:]
        
        for exp in batch:
            # Calculate advantage (how much better/worse than average)
            avg_reward = sum(e.reward for e in batch) / len(batch)
            advantage = exp.reward - avg_reward
            
            # Adjust weights of units that produced these outputs
            self._adjust_weights(advantage)
    
    def _adjust_weights(self, advantage: float):
        """Adjust network weights based on advantage."""
        for unit in self.network.units.values():
            # Randomly adjust some weights
            if random.random() < 0.1:  # 10% of units
                for i in range(len(unit.weights)):
                    # Gradient-like adjustment
                    adjustment = self.learning_rate * advantage * random.uniform(-1, 1)
                    unit.weights[i] += adjustment
                    
                    # Clamp weights
                    unit.weights[i] = max(-10.0, min(10.0, unit.weights[i]))
                
                # Adjust bias
                unit.bias += self.learning_rate * advantage * random.uniform(-0.5, 0.5)
                unit.bias = max(-10.0, min(10.0, unit.bias))
    
    def evolve_rules(self, rules: list, fitness_fn: Callable, 
                     generations: int = 10) -> list:
        """
        Evolve rules using genetic algorithm.
        
        Args:
            rules: List of rules to evolve
            fitness_fn: Function that scores a rule (higher is better)
            generations: Number of evolution cycles
            
        Returns:
            Evolved rules
        """
        population = list(rules)
        
        for gen in range(generations):
            # Score all rules
            scored = [(rule, fitness_fn(rule)) for rule in population]
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # Keep top 50%
            survivors = [rule for rule, score in scored[:len(scored)//2]]
            
            # Generate offspring through mutation
            offspring = []
            for _ in range(len(rules) - len(survivors)):
                parent = random.choice(survivors)
                child = self._mutate_rule(parent)
                offspring.append(child)
            
            population = survivors + offspring
        
        return population
    
    def _mutate_rule(self, rule) -> Any:
        """Mutate a rule (placeholder - would need rule structure)."""
        # For now, just return the rule
        # In a real implementation, we'd modify the rule's condition/action
        return rule
    
    def get_stats(self) -> dict:
        """Get learning statistics."""
        recent_rewards = self.reward_history[-100:] if self.reward_history else [0]
        return {
            "total_experiences": len(self.experiences),
            "total_reward": self.total_reward,
            "avg_recent_reward": sum(recent_rewards) / len(recent_rewards),
            "best_reward": max(self.reward_history) if self.reward_history else 0,
            "worst_reward": min(self.reward_history) if self.reward_history else 0,
        }
