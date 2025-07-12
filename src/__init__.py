"""
Generative Friending Recommendations

This package implements various approaches for friending recommendations,
starting with next target prediction and moving towards generative models.
"""

from .next_target_prediction_userids import NextTargetPredictionUserIDs, NextTargetPredictionBatch

__all__ = ["NextTargetPredictionUserIDs", "NextTargetPredictionBatch"] 