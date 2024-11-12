import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    A class for detecting anomalies in time series data using multiple methods.
    
    Methods include:
    - Statistical (Z-score, IQR)
    - Machine Learning (Isolation Forest)
    - Domain-specific rules
    - Change point detection
    """
    
    def __init__(self, 
                 sensitivity: float = 2.0,
                 contamination: float = 0.1,
                 window_size: int = 7):
        """
        Initialize anomaly detector.
        
        Args:
            sensitivity: Z-score threshold for statistical detection
            contamination: Expected proportion of outliers in the dataset
            window_size: Rolling window size for dynamic thresholds
        """
        self.sensitivity = sensitivity
        self.contamination = contamination
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        logger.info("AnomalyDetector initialized successfully")

    def detect_anomalies(self, data: pd.Series) -> Dict[str, Any]:
        """
        Detect anomalies using multiple methods.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary containing anomalies detected by each method
        """
        try:
            logger.info("Starting anomaly detection")
            
            results = {
                'statistical': self._statistical_anomalies(data),
                'isolation_forest': self._isolation_forest_anomalies(data),
                'domain_rules': self._domain_specific_anomalies(data),
                'change_points': self._detect_change_points(data),
                'summary': self._create_anomaly_summary(data)
            }
            
            logger.info("Completed anomaly detection")
            return results
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            raise

    def _statistical_anomalies(self, data: pd.Series) -> Dict[str, pd.Series]:
        """
        Detect anomalies using statistical methods.
        """
        try:
            # Z-score method
            z_scores = stats.zscore(data)
            z_score_anomalies = data[abs(z_scores) > self.sensitivity]
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            iqr_anomalies = data[
                (data < (Q1 - 1.5 * IQR)) | 
                (data > (Q3 + 1.5 * IQR))
            ]
            
            # Moving average method
            rolling_mean = data.rolling(window=self.window_size).mean()
            rolling_std = data.rolling(window=self.window_size).std()
            moving_avg_anomalies = data[
                abs(data - rolling_mean) > (self.sensitivity * rolling_std)
            ]
            
            return {
                'z_score': z_score_anomalies,
                'iqr': iqr_anomalies,
                'moving_average': moving_avg_anomalies
            }
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {str(e)}")
            raise

    def _isolation_forest_anomalies(self, data: pd.Series) -> pd.Series:
        """
        Detect anomalies using Isolation Forest.
        """
        try:
            # Prepare data
            X = data.values.reshape(-1, 1)
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit and predict
            self.isolation_forest.fit(X_scaled)
            predictions = self.isolation_forest.predict(X_scaled)
            
            # Return anomalies
            return data[predictions == -1]
        except Exception as e:
            logger.error(f"Error in Isolation Forest anomaly detection: {str(e)}")
            raise

    def _domain_specific_anomalies(self, data: pd.Series) -> Dict[str, pd.Series]:
        """
        Detect anomalies based on domain-specific rules.
        """
        try:
            # Calculate daily changes
            daily_changes = data.pct_change()
            
            # Define rules
            sudden_spikes = data[daily_changes > 0.5]  # 50% increase
            sudden_drops = data[daily_changes < -0.5]  # 50% decrease
            zero_demand = data[data == 0]  # Zero demand
            excessive_demand = data[data > data.mean() + 3 * data.std()]
            
            # Detect repeated patterns
            pattern_anomalies = self._detect_pattern_anomalies(data)
            
            return {
                'sudden_spikes': sudden_spikes,
                'sudden_drops': sudden_drops,
                'zero_demand': zero_demand,
                'excessive_demand': excessive_demand,
                'pattern_anomalies': pattern_anomalies
            }
        except Exception as e:
            logger.error(f"Error in domain-specific anomaly detection: {str(e)}")
            raise

    def _detect_pattern_anomalies(self, data: pd.Series) -> pd.Series:
        """
        Detect unusual patterns in the time series.
        """
        try:
            # Calculate weekly patterns
            weekly_pattern = data.groupby(data.index.dayofweek).mean()
            weekly_std = data.groupby(data.index.dayofweek).std()
            
            pattern_anomalies = pd.Series(index=data.index, dtype=bool)
            
            for day in range(7):
                day_data = data[data.index.dayofweek == day]
                day_mean = weekly_pattern[day]
                day_std = weekly_std[day]
                
                pattern_anomalies[day_data.index] = abs(
                    day_data - day_mean
                ) > (self.sensitivity * day_std)
            
            return data[pattern_anomalies]
        except Exception as e:
            logger.error(f"Error in pattern anomaly detection: {str(e)}")
            raise

    def _detect_change_points(self, data: pd.Series) -> List[datetime]:
        """
        Detect significant changes in the time series.
        """
        try:
            # Calculate rolling statistics
            rolling_mean = data.rolling(window=self.window_size).mean()
            rolling_std = data.rolling(window=self.window_size).std()
            
            # Detect significant changes
            mean_changes = rolling_mean.diff().abs()
            significant_changes = data[
                mean_changes > (self.sensitivity * rolling_std)
            ].index.tolist()
            
            return significant_changes
        except Exception as e:
            logger.error(f"Error in change point detection: {str(e)}")
            raise

    def _create_anomaly_summary(self, data: pd.Series) -> Dict[str, Any]:
        """
        Create a summary of all detected anomalies.
        """
        try:
            total_points = len(data)
            statistical_anomalies = self._statistical_anomalies(data)
            domain_anomalies = self._domain_specific_anomalies(data)
            
            summary = {
                'total_points': total_points,
                'anomaly_counts': {
                    'statistical': {
                        'z_score': len(statistical_anomalies['z_score']),
                        'iqr': len(statistical_anomalies['iqr']),
                        'moving_average': len(statistical_anomalies['moving_average'])
                    },
                    'isolation_forest': len(self._isolation_forest_anomalies(data)),
                    'domain_rules': {
                        'sudden_spikes': len(domain_anomalies['sudden_spikes']),
                        'sudden_drops': len(domain_anomalies['sudden_drops']),
                        'zero_demand': len(domain_anomalies['zero_demand']),
                        'excessive_demand': len(domain_anomalies['excessive_demand'])
                    }
                },
                'change_points': len(self._detect_change_points(data))
            }
            
            # Calculate percentages
            summary['anomaly_percentages'] = {
                method: (count / total_points) * 100
                for method, count in summary['anomaly_counts'].items()
                if isinstance(count, (int, float))
            }
            
            return summary
        except Exception as e:
            logger.error(f"Error in creating anomaly summary: {str(e)}")

class AnomalyAlert:
    """
    A class for generating alerts based on detected anomalies.
    """
    
    def __init__(self, threshold_config: Optional[Dict[str, float]] = None):
        """
        Initialize alert system.
        
        Args:
            threshold_config: Custom threshold configuration
        """
        self.threshold_config = threshold_config or {
            'sudden_spike': 0.5,      # 50% increase
            'sudden_drop': -0.5,      # 50% decrease
            'sustained_change': 0.3,   # 30% change over 3 days
            'zero_demand_days': 3,     # Alert after 3 days of zero demand
            'excessive_demand': 3.0    # 3 standard deviations above mean
        }
        logger.info("AnomalyAlert system initialized successfully")

    def generate_alerts(self, 
                       historical_data: pd.Series, 
                       forecast_data: Optional[pd.Series] = None) -> List[Dict[str, Any]]:
        """
        Generate alerts based on historical and forecast data.
        
        Args:
            historical_data: Historical time series data
            forecast_data: Optional forecast data
            
        Returns:
            List of alert dictionaries
        """
        try:
            logger.info("Starting alert generation")
            alerts = []
            
            # Historical data alerts
            alerts.extend(self._check_historical_patterns(historical_data))
            
            # Forecast alerts if available
            if forecast_data is not None:
                alerts.extend(self._check_forecast_anomalies(
                    historical_data, forecast_data
                ))
            
            # Sort alerts by severity and date
            alerts.sort(key=lambda x: (
                {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[x['severity']], 
                x['date']
            ))
            
            logger.info(f"Generated {len(alerts)} alerts")
            return alerts
        except Exception as e:
            logger.error(f"Error in alert generation: {str(e)}")
            raise

    def _check_historical_patterns(self, data: pd.Series) -> List[Dict[str, Any]]:
        """
        Check for anomalies in historical patterns.
        """
        alerts = []
        try:
            # Check for consecutive zero demand
            zero_demand_streak = 0
            last_zero_date = None
            
            for date, value in data.items():
                if value == 0:
                    zero_demand_streak += 1
                    last_zero_date = date
                    if zero_demand_streak >= self.threshold_config['zero_demand_days']:
                        alerts.append({
                            'type': 'ZERO_DEMAND',
                            'date': date,
                            'value': 0,
                            'streak': zero_demand_streak,
                            'severity': 'HIGH',
                            'message': f'No demand for {zero_demand_streak} consecutive days'
                        })
                else:
                    zero_demand_streak = 0
            
            # Check for sustained changes
            rolling_change = data.pct_change(periods=3).rolling(window=3).mean()
            sustained_changes = rolling_change[
                abs(rolling_change) > self.threshold_config['sustained_change']
            ]
            
            for date, change in sustained_changes.items():
                alerts.append({
                    'type': 'SUSTAINED_CHANGE',
                    'date': date,
                    'change': change,
                    'severity': 'MEDIUM',
                    'message': f'Sustained {"increase" if change > 0 else "decrease"} '
                             f'of {abs(change):.1%} over 3 days'
                })
            
            # Check for excessive demand
            mean_demand = data.mean()
            std_demand = data.std()
            threshold = mean_demand + (
                self.threshold_config['excessive_demand'] * std_demand
            )
            
            excessive = data[data > threshold]
            for date, value in excessive.items():
                alerts.append({
                    'type': 'EXCESSIVE_DEMAND',
                    'date': date,
                    'value': value,
                    'severity': 'HIGH',
                    'message': f'Unusually high demand: {value:.0f} '
                             f'(threshold: {threshold:.0f})'
                })
            
            return alerts
        except Exception as e:
            logger.error(f"Error in historical pattern checking: {str(e)}")
            raise

    def _check_forecast_anomalies(self, 
                                historical_data: pd.Series,
                                forecast_data: pd.Series) -> List[Dict[str, Any]]:
        """
        Check for anomalies in forecast data.
        """
        alerts = []
        try:
            hist_mean = historical_data.mean()
            hist_std = historical_data.std()
            
            # Check each forecasted point
            for date, value in forecast_data.items():
                # Sudden spike
                if value > hist_mean + (hist_std * self.threshold_config['excessive_demand']):
                    alerts.append({
                        'type': 'FORECAST_SPIKE',
                        'date': date,
                        'value': value,
                        'threshold': hist_mean + (hist_std * self.threshold_config['excessive_demand']),
                        'severity': 'HIGH',
                        'message': f'Unusual high demand forecasted: {value:.0f} '
                                 f'(threshold: {hist_mean + hist_std * self.threshold_config["excessive_demand"]:.0f})'
                    })
                
                # Sudden drop
                if value < hist_mean - (hist_std * self.threshold_config['excessive_demand']):
                    alerts.append({
                        'type': 'FORECAST_DROP',
                        'date': date,
                        'value': value,
                        'threshold': hist_mean - (hist_std * self.threshold_config['excessive_demand']),
                        'severity': 'HIGH',
                        'message': f'Unusual low demand forecasted: {value:.0f} '
                                 f'(threshold: {hist_mean - hist_std * self.threshold_config["excessive_demand"]:.0f})'
                    })
            
            return alerts
        except Exception as e:
            logger.error(f"Error in forecast anomaly checking: {str(e)}")
            raise

    def get_alert_summary(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary of generated alerts.
        """
        try:
            summary = {
                'total_alerts': len(alerts),
                'by_severity': {
                    'HIGH': len([a for a in alerts if a['severity'] == 'HIGH']),
                    'MEDIUM': len([a for a in alerts if a['severity'] == 'MEDIUM']),
                    'LOW': len([a for a in alerts if a['severity'] == 'LOW'])
                },
                'by_type': {}
            }
            
            # Count alerts by type
            for alert in alerts:
                alert_type = alert['type']
                if alert_type not in summary['by_type']:
                    summary['by_type'][alert_type] = 0
                summary['by_type'][alert_type] += 1
            
            return summary
        except Exception as e:
            logger.error(f"Error in creating alert summary: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Create sample data with intentional anomalies
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Base demand with seasonality
    t = np.arange(len(dates))
    base_demand = 100 + 20 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
    
    # Add anomalies
    demand = base_demand.copy()
    demand[50:53] = 0  # Zero demand period
    demand[100] = 500  # Sudden spike
    demand[150:153] *= 2  # Sustained increase
    demand[200] = 10  # Sudden drop
    
    # Add noise
    demand += np.random.normal(0, 10, len(dates))
    demand = np.maximum(demand, 0)  # Ensure non-negative values
    
    # Create time series
    historical_data = pd.Series(demand, index=dates)
    
    # Initialize detectors
    detector = AnomalyDetector()
    alert_system = AnomalyAlert()
    
    # Detect anomalies and generate alerts
    anomalies = detector.detect_anomalies(historical_data)
    alerts = alert_system.generate_alerts(historical_data)
    
    # Print results
    print("\nAnomaly Detection Results:")
    print(json.dumps(anomalies['summary'], indent=2))
    
    print("\nGenerated Alerts:")
    for alert in alerts:
        print(f"\nType: {alert['type']}")
        print(f"Date: {alert['date']}")
        print(f"Severity: {alert['severity']}")
        print(f"Message: {alert['message']}")