"""VENOM-PARASITE: training-free middleware that gives an LLM tool access to its own internals.

The host model is the agent. VENOM knows nothing on its own - it only serves tools.
Every intervention is measured, bounded and reversible.
"""

__version__ = "0.1.1"
