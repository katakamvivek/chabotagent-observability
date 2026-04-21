"""
Pytest configuration — ensures the project root is on sys.path
and sets the OpenAI key for DeepEval's LLM judge.
"""

import os
import sys
from pathlib import Path

# Project root on path so `from agents import run` works
sys.path.insert(0, str(Path(__file__).parent.parent))

# DeepEval uses OpenAI as its judge LLM — point it to the same key
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
