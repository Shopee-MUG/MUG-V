#!/usr/bin/env python3
"""
Text encoder module.

Provides T5-based text encoding for conditioning video generation.
"""

from .t5 import T5Encoder as T5

__all__ = ["T5"]