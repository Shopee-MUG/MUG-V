#!/usr/bin/env python3
"""Setup script for MUG-DiT-10B."""

import subprocess
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("--"):
            requirements.append(line)

setup(
    name="mug_v",
    version="0.1.0",
    author="MUG-DiT Team",
    author_email="mug-dit-team@example.com",
    description="MUG-DiT-10B: Multi-scale Unified Generation Diffusion Transformer for Video Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Shopee-MUG/MUG-V",
    project_urls={
        "Bug Reports": "https://github.com/Shopee-MUG/MUG-V/issues",
        "Source": "https://github.com/Shopee-MUG/MUG-V",
        "Documentation": "https://github.com/Shopee-MUG/MUG-V#readme",
    },
    packages=find_packages(
        exclude=(
            "assets",
            "docs",
            "eval",
            "samples",
            "gradio",
            "logs",
            "outputs",
            "scripts",
            "*.egg-info",
        )
    ),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    zip_safe=False,
    keywords="video generation, diffusion models, transformers, AI, machine learning",
)
