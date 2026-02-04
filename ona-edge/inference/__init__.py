"""
ONA Edge AI Inference Module
Runs diagnostic models on medical images
"""

from .engine import InferenceEngine, get_engine

__all__ = ['InferenceEngine', 'get_engine']
