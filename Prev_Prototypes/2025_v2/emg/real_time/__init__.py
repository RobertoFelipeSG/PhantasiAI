"""
Real-time EMG processing module.

This module contains components for live EMG data acquisition,
recording, and real-time analysis.
"""

from .real_time_recorder import RealTimeRecorder
from .real_time_peak_analyzer import RealTimePeakAnalyzer

__all__ = ['RealTimeRecorder', 'RealTimePeakAnalyzer']
