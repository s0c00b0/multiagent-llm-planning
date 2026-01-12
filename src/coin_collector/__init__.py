"""
Coin Collector Game - A Python implementation of the Coin Collector benchmark.

This module provides a text-based game where players must navigate through
locations and collect all coins to win.
"""

from .game import CoinCollectorGame, Direction
from .visualize import visualize_map

__all__ = ['CoinCollectorGame', 'Direction', 'visualize_map']
__version__ = '1.0.0'

