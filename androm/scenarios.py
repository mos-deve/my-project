"""
Scenarios - Complex test scenarios for ANDROM.
Real coding challenges to test and improve the system.
"""

from __future__ import annotations
import ast
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TestCase:
    """A single test case."""
    inputs: tuple
    expected: Any
    description: str = ""


@dataclass
class Scenario:
    """A coding challenge scenario."""
    name: str
    description: str
    test_cases: list[TestCase]
    reference_solution: str
    difficulty: int  # 1-10
    category: str
    
    def evaluate(self, solution_code: str) -> dict:
        """
        Evaluate a solution against all test cases.
        
        Returns:
            Dictionary with score, passed, failed, errors
        """
        passed = 0
        failed = 0
        errors = []
        
        # Check syntax
        try:
            ast.parse(solution_code)
        except SyntaxError as e:
            return {
                "score": 0.0,
                "passed": 0,
                "failed": len(self.test_cases),
                "errors": [f"Syntax error: {e}"],
                "valid": False,
            }
        
        # Execute solution
        namespace = {}
        try:
            exec(solution_code, namespace)
        except Exception as e:
            return {
                "score": 0.0,
                "passed": 0,
                "failed": len(self.test_cases),
                "errors": [f"Execution error: {e}"],
                "valid": False,
            }
        
        # Find the main function
        func_name = None
        for name in namespace:
            if callable(namespace[name]) and not name.startswith("_"):
                func_name = name
                break
        
        if not func_name:
            return {
                "score": 0.0,
                "passed": 0,
                "failed": len(self.test_cases),
                "errors": ["No callable function found"],
                "valid": False,
            }
        
        func = namespace[func_name]
        
        # Run test cases
        for tc in self.test_cases:
            try:
                if isinstance(tc.inputs, tuple):
                    result = func(*tc.inputs)
                else:
                    result = func(tc.inputs)
                
                if result == tc.expected:
                    passed += 1
                else:
                    failed += 1
                    errors.append(f"Expected {tc.expected}, got {result}")
            except Exception as e:
                failed += 1
                errors.append(f"Error: {e}")
        
        total = len(self.test_cases)
        score = (passed / total * 100) if total > 0 else 0
        
        # Bonus for shorter solutions
        solution_lines = len(solution_code.strip().splitlines())
        if solution_lines < len(self.reference_solution.strip().splitlines()):
            score *= 1.1  # 10% bonus for shorter code
        
        return {
            "score": min(score, 100.0),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "valid": True,
            "solution_lines": solution_lines,
        }


class ScenarioLibrary:
    """Library of coding challenges."""
    
    def __init__(self):
        self.scenarios: list[Scenario] = []
        self._init_scenarios()
    
    def _init_scenarios(self):
        """Initialize with challenge scenarios."""
        
        # 1. FizzBuzz
        self.scenarios.append(Scenario(
            name="fizzbuzz",
            description="Return 'Fizz' if divisible by 3, 'Buzz' if divisible by 5, 'FizzBuzz' if both, else the number as string",
            test_cases=[
                TestCase((3,), "Fizz"),
                TestCase((5,), "Buzz"),
                TestCase((15,), "FizzBuzz"),
                TestCase((7,), "7"),
                TestCase((30,), "FizzBuzz"),
            ],
            reference_solution="""def fizzbuzz(n):
    if n % 15 == 0:
        return "FizzBuzz"
    elif n % 3 == 0:
        return "Fizz"
    elif n % 5 == 0:
        return "Buzz"
    return str(n)""",
            difficulty=2,
            category="logic",
        ))
        
        # 2. Fibonacci
        self.scenarios.append(Scenario(
            name="fibonacci",
            description="Return the nth Fibonacci number (0-indexed)",
            test_cases=[
                TestCase((0,), 0),
                TestCase((1,), 1),
                TestCase((5,), 5),
                TestCase((10,), 55),
                TestCase((15,), 610),
            ],
            reference_solution="""def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b""",
            difficulty=3,
            category="math",
        ))
        
        # 3. Palindrome check
        self.scenarios.append(Scenario(
            name="is_palindrome",
            description="Check if a string is a palindrome (case-insensitive, ignore spaces)",
            test_cases=[
                TestCase(("racecar",), True),
                TestCase(("hello",), False),
                TestCase(("A man a plan a canal Panama",), True),
                TestCase(("",), True),
                TestCase(("ab",), False),
            ],
            reference_solution="""def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]""",
            difficulty=2,
            category="strings",
        ))
        
        # 4. Flatten nested list
        self.scenarios.append(Scenario(
            name="flatten",
            description="Flatten a nested list into a single list",
            test_cases=[
                TestCase(([1, [2, 3], 4],), [1, 2, 3, 4]),
                TestCase(([[1, 2], [3, [4, 5]]],), [1, 2, 3, 4, 5]),
                TestCase(([1, 2, 3],), [1, 2, 3]),
                TestCase(([],), []),
                TestCase(([[[[1]]]],), [1]),
            ],
            reference_solution="""def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result""",
            difficulty=4,
            category="recursion",
        ))
        
        # 5. Binary search
        self.scenarios.append(Scenario(
            name="binary_search",
            description="Find index of target in sorted list, return -1 if not found",
            test_cases=[
                TestCase(([1, 2, 3, 4, 5], 3), 2),
                TestCase(([1, 2, 3, 4, 5], 1), 0),
                TestCase(([1, 2, 3, 4, 5], 5), 4),
                TestCase(([1, 2, 3, 4, 5], 6), -1),
                TestCase(([], 1), -1),
            ],
            reference_solution="""def binary_search(lst, target):
    lo, hi = 0, len(lst) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if lst[mid] == target:
            return mid
        elif lst[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1""",
            difficulty=5,
            category="algorithms",
        ))
        
        # 6. Matrix multiplication
        self.scenarios.append(Scenario(
            name="matrix_multiply",
            description="Multiply two 2D matrices",
            test_cases=[
                TestCase(([[1, 2], [3, 4]], [[5, 6], [7, 8]]), [[19, 22], [43, 50]]),
                TestCase(([[1]], [[2]]), [[2]]),
                TestCase(([[1, 0], [0, 1]], [[5, 6], [7, 8]]), [[5, 6], [7, 8]]),
            ],
            reference_solution="""def matrix_multiply(a, b):
    rows_a, cols_a = len(a), len(a[0])
    cols_b = len(b[0])
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result""",
            difficulty=6,
            category="math",
        ))
        
        # 7. Caesar cipher
        self.scenarios.append(Scenario(
            name="caesar_cipher",
            description="Encrypt text using Caesar cipher with given shift",
            test_cases=[
                TestCase(("abc", 1), "bcd"),
                TestCase(("xyz", 3), "abc"),
                TestCase(("Hello World!", 5), "Mjqqt Btwqi!"),
                TestCase(("", 5), ""),
            ],
            reference_solution="""def caesar_cipher(text, shift):
    result = []
    for c in text:
        if c.isalpha():
            base = ord('A') if c.isupper() else ord('a')
            result.append(chr((ord(c) - base + shift) % 26 + base))
        else:
            result.append(c)
    return ''.join(result)""",
            difficulty=4,
            category="strings",
        ))
        
        # 8. Merge sorted lists
        self.scenarios.append(Scenario(
            name="merge_sorted",
            description="Merge two sorted lists into one sorted list",
            test_cases=[
                TestCase(([1, 3, 5], [2, 4, 6]), [1, 2, 3, 4, 5, 6]),
                TestCase(([1, 2, 3], []), [1, 2, 3]),
                TestCase(([], []), []),
                TestCase(([1], [2]), [1, 2]),
            ],
            reference_solution="""def merge_sorted(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result""",
            difficulty=4,
            category="algorithms",
        ))
        
        # 9. Prime sieve
        self.scenarios.append(Scenario(
            name="sieve",
            description="Return list of prime numbers up to n",
            test_cases=[
                TestCase((10,), [2, 3, 5, 7]),
                TestCase((20,), [2, 3, 5, 7, 11, 13, 17, 19]),
                TestCase((2,), [2]),
                TestCase((1,), []),
            ],
            reference_solution="""def sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]""",
            difficulty=5,
            category="math",
        ))
        
        # 10. LRU Cache
        self.scenarios.append(Scenario(
            name="lru_cache",
            description="Implement an LRU cache with get and put operations",
            test_cases=[
                TestCase(([
                    ("put", 1, 1),
                    ("put", 2, 2),
                    ("get", 1, 1),
                    ("put", 3, 3),
                    ("get", 2, -1),
                    ("put", 4, 4),
                    ("get", 1, -1),
                    ("get", 3, 3),
                    ("get", 4, 4),
                ],), True),
            ],
            reference_solution="""def lru_cache(operations):
    from collections import OrderedDict
    cache = OrderedDict()
    capacity = 2
    results = []
    for op in operations:
        if op[0] == "put":
            key, val = op[1], op[2]
            if key in cache:
                del cache[key]
            cache[key] = val
            if len(cache) > capacity:
                cache.popitem(last=False)
        elif op[0] == "get":
            key = op[1]
            expected = op[2]
            if key in cache:
                cache.move_to_end(key)
                results.append(cache[key] == expected)
            else:
                results.append(-1 == expected)
    return all(results)""",
            difficulty=7,
            category="data_structures",
        ))
    
    def get_by_difficulty(self, min_diff: int = 1, max_diff: int = 10) -> list[Scenario]:
        """Get scenarios within difficulty range."""
        return [s for s in self.scenarios if min_diff <= s.difficulty <= max_diff]
    
    def get_by_category(self, category: str) -> list[Scenario]:
        """Get scenarios by category."""
        return [s for s in self.scenarios if s.category == category]
    
    def get_random(self) -> Scenario:
        """Get a random scenario."""
        import random
        return random.choice(self.scenarios)
