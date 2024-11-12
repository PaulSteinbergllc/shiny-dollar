"""
Pharmacy Analytics Package

This package provides tools for analyzing pharmacy inventory data, including:
- Trend analysis
- Anomaly detection
- Time series forecasting
"""

from .trend_analyzer import TrendAnalyzer
from .anomaly_detector import AnomalyDetector, AnomalyAlert

# Version of the analysis package
__version__ = '0.1.0'

# List of public classes/functions that should be available when using "from analysis import *"
__all__ = [
    'TrendAnalyzer',
    'AnomalyDetector',
    'AnomalyAlert'
]

# Optional: Create convenient factory functions
def create_trend_analyzer(data):
    """
    Convenience function to create a TrendAnalyzer instance
    
    Args:
        data (pd.Series): Time series data to analyze
        
    Returns:
        TrendAnalyzer: Initialized trend analyzer
    """
    return TrendAnalyzer(data)

def create_anomaly_detector(sensitivity=2.0):
    """
    Convenience function to create an AnomalyDetector instance
    
    Args:
        sensitivity (float): Sensitivity threshold for anomaly detection
        
    Returns:
        AnomalyDetector: Initialized anomaly detector
    """
    return AnomalyDetector(sensitivity=sensitivity)

def create_alert_system(threshold_config=None):
    """
    Convenience function to create an AnomalyAlert instance
    
    Args:
        threshold_config (dict, optional): Custom threshold configuration
        
    Returns:
        AnomalyAlert: Initialized alert system
    """
    return AnomalyAlert(threshold_config=threshold_config)

# Optional: Package metadata
__author__ = 'Your Name'
__email__ = 'your.email@example.com'
__description__ = 'A package for pharmacy inventory analysis and forecasting' 