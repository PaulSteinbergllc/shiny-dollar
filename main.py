from analysis import (
    TrendAnalyzer,
    AnomalyDetector,
    AnomalyAlert,
    create_trend_analyzer,
    create_anomaly_detector,
    create_alert_system
)
from your_existing_code import InventoryForecastAnalyzer  # Your existing forecasting code

class PharmacyAnalytics:
    def __init__(self):
        self.forecast_analyzer = None
        self.trend_analyzer = None
        self.anomaly_detector = None
        self.alert_system = None

    def initialize_with_data(self, historical_data: pd.DataFrame):
        """Initialize all analyzers with historical data"""
        self.forecast_analyzer = InventoryForecastAnalyzer(historical_data)
        self.trend_analyzer = create_trend_analyzer(historical_data.set_index('date')['quantity'])
        self.anomaly_detector = create_anomaly_detector(sensitivity=2.5)
        self.alert_system = create_alert_system()

    def run_complete_analysis(self, medication_id: str):
        """Run all analyses for a given medication"""
        # Run forecasting
        forecast_results = self.forecast_analyzer.generate_forecasts(medication_id)
        
        # Run trend analysis
        trend_results = self.trend_analyzer.analyze_all_trends()
        
        # Run anomaly detection
        anomaly_results = self.anomaly_detector.detect_anomalies(
            self.trend_analyzer.data
        )
        
        # Generate alerts
        alerts = self.alert_system.generate_alerts(
            self.trend_analyzer.data,
            forecast_results['forecast']
        )
        
        return {
            'forecasting': forecast_results,
            'trends': trend_results,
            'anomalies': anomaly_results,
            'alerts': alerts
        }

    def visualize_results(self, medication_id: str):
        """Create visualizations for all analyses"""
        # Forecast visualization
        self.forecast_analyzer.plot_comprehensive_analysis(medication_id)
        
        # Trend visualization
        self.trend_analyzer.plot_comprehensive_analysis()

def main():
    # Example usage
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate sample data
    t = np.arange(len(dates))
    trend = 0.1 * t
    seasonal_weekly = 20 * np.sin(2 * np.pi * t / 7)
    seasonal_yearly = 50 * np.sin(2 * np.pi * t / 365)
    noise = np.random.normal(0, 10, len(dates))
    
    data = 100 + trend + seasonal_weekly + seasonal_yearly + noise
    sample_data = pd.DataFrame({
        'date': dates,
        'medication_id': 'MED001',
        'quantity': data
    })
    
    # Initialize and run analysis
    analytics = PharmacyAnalytics()
    analytics.initialize_with_data(sample_data)
    
    # Run complete analysis for a medication
    results = analytics.run_complete_analysis('MED001')
    
    # Visualize results
    analytics.visualize_results('MED001')

if __name__ == "__main__":
    main()
