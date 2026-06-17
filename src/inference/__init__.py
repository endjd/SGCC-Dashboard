"""
Inference pipeline module for production predictions.
"""

from .pipeline import InferencePipeline, predict_customer_theft

__all__ = ["InferencePipeline", "predict_customer_theft"]
