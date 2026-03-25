"""
CLI entry point for ANDROM.
"""

import argparse
import json
import sys
from androm import Brain, __version__


def main():
    parser = argparse.ArgumentParser(
        prog="androm",
        description="ANDROM - Adaptive Network of Deterministic Rule-based Operations and Mathematics",
    )
    parser.add_argument("--version", action="version", version=f"androm {__version__}")
    parser.add_argument("--units", type=int, default=1000, help="Number of computational units")
    parser.add_argument("--connectivity", type=float, default=0.05, help="Network connectivity (0-1)")
    parser.add_argument("--cycles", type=int, default=10, help="Number of thinking cycles")
    parser.add_argument("--optimize", action="store_true", help="Run self-optimization")
    parser.add_argument("--status", action="store_true", help="Show brain status")
    
    args = parser.parse_args()
    
    print(f"ANDROM v{__version__}")
    print(f"Building brain with {args.units} units...")
    
    brain = Brain()
    brain.build_network(num_units=args.units, connectivity=args.connectivity)
    
    if args.status:
        status = brain.status()
        print(json.dumps(status, indent=2))
        return
    
    if args.optimize:
        print("\nRunning self-optimization...")
        results = brain.optimize_self()
        print(json.dumps(results, indent=2))
        return
    
    print(f"\nRunning {args.cycles} thinking cycles...\n")
    
    for i in range(args.cycles):
        result = brain.run_cycle()
        print(f"Cycle {result['cycle']}: {result['rules_fired']} rules fired, {result['code_lines']} code lines")
    
    print(f"\n--- Brain Status ---")
    status = brain.status()
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
