#!/usr/bin/env python3
"""
Scheduler module for MUG-DiT-10B inference.

Provides sampling schedulers for the diffusion process.
"""

from .rectified_flow import RFlow

__all__ = ["RFlow"]