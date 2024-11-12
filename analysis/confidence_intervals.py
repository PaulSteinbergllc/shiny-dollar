import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
import logging
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfidenceIntervalCalculator:
    """
    A class for calculating and analyzing confidence intervals for forecasts.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize confidence interval calculator.
        
        Args:
            confidence_level: Desired confidence level (default: 0.95 for 95% CI)
        """
        self.confidence_level = confidence_level
        self.z_score = stats.norm.ppf((1 + confidence_level) / 2)
        logger.info(f"Initialized CI calculator with {confidence_level*100}% confidence level")

    def calculate_confidence_intervals(self, 
                                    forecast: pd.Series,
                                    historical_data: pd.Series,
                                    prediction_errors: Optional[pd.Series] = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate comprehensive confidence intervals using multiple methods.
        
        Args:
            forecast: Forecasted values
            historical_data: Historical time series data
            prediction_errors: Optional historical prediction errors
            
        Returns:
            Dictionary containing different types of confidence intervals
        """
        try:
            logger.info("Calculating confidence intervals")
            
            results = {
                'standard_ci': self._calculate_standard_ci(forecast, historical_data),
                'bootstrap_ci': self._calculate_bootstrap_ci(forecast, historical_data),
                'dynamic_ci': self._calculate_dynamic_ci(forecast, historical_data),
                'empirical_ci': self._calculate_empirical_ci(forecast, prediction_errors)
                if prediction_errors is not None else None
            }
            
            # Add combined CI that considers all methods
            results['combined_ci'] = self._combine_confidence_intervals(results)
            
            # Add CI metrics and analysis
            results['metrics'] = self._calculate_ci_metrics(results)
            
            logger.info("Completed confidence interval calculations")
            return results
        except Exception as e:
            logger.error(f"Error in confidence interval calculation: {str(e)}")
            raise

    def _calculate_standard_ci(self, 
                             forecast: pd.Series,
                             historical_data: pd.Series) -> pd.DataFrame:
        """
        Calculate standard confidence intervals based on historical variance.
        """
        try:
            # Calculate standard error
            std_error = np.std(historical_data)
            forecast_std = std_error * np.sqrt(1 + np.arange(len(forecast)) / len(historical_data))
            
            ci_lower = forecast - (self.z_score * forecast_std)
            ci_upper = forecast + (self.z_score * forecast_std)
            
            return pd.DataFrame({
                'forecast': forecast,
                'lower_bound': ci_lower,
                'upper_bound': ci_upper,
                'std_error': forecast_std
            })
        except Exception as e:
            logger.error(f"Error in standard CI calculation: {str(e)}")
            raise

    def _calculate_bootstrap_ci(self,
                              forecast: pd.Series,
                              historical_data: pd.Series,
                              n_bootstrap: int = 1000) -> pd.DataFrame:
        """
        Calculate confidence intervals using bootstrap method.
        """
        try:
            bootstrap_forecasts = []
            
            for _ in range(n_bootstrap):
                # Resample historical data
                bootstrap_sample = historical_data.sample(
                    n=len(historical_data),
                    replace=True
                )
                
                # Calculate mean and std of bootstrap sample
                bootstrap_std = np.std(bootstrap_sample)
                
                # Generate bootstrap forecast
                bootstrap_forecast = forecast + np.random.normal(
                    0, bootstrap_std, size=len(forecast)
                )
                bootstrap_forecasts.append(bootstrap_forecast)
            
            # Calculate percentile-based CI
            bootstrap_forecasts = np.array(bootstrap_forecasts)
            lower_percentile = (1 - self.confidence_level) / 2
            upper_percentile = 1 - lower_percentile
            
            ci_lower = np.percentile(bootstrap_forecasts, lower_percentile * 100, axis=0)
            ci_upper = np.percentile(bootstrap_forecasts, upper_percentile * 100, axis=0)
            
            return pd.DataFrame({
                'forecast': forecast,
                'lower_bound': ci_lower,
                'upper_bound': ci_upper,
                'std_error': np.std(bootstrap_forecasts, axis=0)
            })
        except Exception as e:
            logger.error(f"Error in bootstrap CI calculation: {str(e)}")
            raise

    def _calculate_dynamic_ci(self,
                            forecast: pd.Series,
                            historical_data: pd.Series,
                            window_size: int = 30) -> pd.DataFrame:
        """
        Calculate dynamic confidence intervals based on recent volatility.
        """
        try:
            # Calculate rolling standard deviation
            rolling_std = historical_data.rolling(window=window_size).std()
            last_rolling_std = rolling_std.iloc[-1]
            
            # Adjust CI width based on forecast horizon
            horizon_factor = 1 + np.log1p(np.arange(len(forecast)) / window_size)
            forecast_std = last_rolling_std * horizon_factor
            
            ci_lower = forecast - (self.z_score * forecast_std)
            ci_upper = forecast + (self.z_score * forecast_std)
            
            return pd.DataFrame({
                'forecast': forecast,
                'lower_bound': ci_lower,
                'upper_bound': ci_upper,
                'std_error': forecast_std
            })
        except Exception as e:
            logger.error(f"Error in dynamic CI calculation: {str(e)}")
            raise

    def _calculate_empirical_ci(self,
                              forecast: pd.Series,
                              prediction_errors: pd.Series) -> pd.DataFrame:
        """
        Calculate confidence intervals based on empirical prediction errors.
        """
        try:
            # Calculate error quantiles
            error_lower = np.percentile(prediction_errors, 
                                      ((1 - self.confidence_level) / 2) * 100)
            error_upper = np.percentile(prediction_errors,
                                      (1 - (1 - self.confidence_level) / 2) * 100)
            
            # Calculate CI
            ci_lower = forecast + error_lower
            ci_upper = forecast + error_upper
            
            return pd.DataFrame({
                'forecast': forecast,
                'lower_bound': ci_lower,
                'upper_bound': ci_upper,
                'std_error': np.std(prediction_errors)
            })
        except Exception as e:
            logger.error(f"Error in empirical CI calculation: {str(e)}")
            raise

    def _combine_confidence_intervals(self, 
                                   ci_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine different CI methods into a single consensus interval.
        """
        try:
            # Collect all lower and upper bounds
            all_lower_bounds = []
            all_upper_bounds = []
            
            for method, ci_df in ci_results.items():
                if method not in ['metrics', 'combined_ci'] and ci_df is not None:
                    all_lower_bounds.append(ci_df['lower_bound'])
                    all_upper_bounds.append(ci_df['upper_bound'])
            
            # Calculate weighted average of bounds
            # Weight could be based on historical performance
            combined_lower = np.mean(all_lower_bounds, axis=0)
            combined_upper = np.mean(all_upper_bounds, axis=0)
            
            return pd.DataFrame({
                'forecast': ci_results['standard_ci']['forecast'],
                'lower_bound': combined_lower,
                'upper_bound': combined_upper,
                'std_error': (combined_upper - combined_lower) / (2 * self.z_score)
            })
        except Exception as e:
            logger.error(f"Error in combining CIs: {str(e)}")
            raise

    def _calculate_ci_metrics(self, ci_results: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate metrics for confidence interval quality.
        """
        try:
            metrics = {
                'average_width': {},
                'asymmetry': {},
                'relative_width': {}
            }
            
            for method, ci_df in ci_results.items():
                if method not in ['metrics', 'combined_ci'] and ci_df is not None:
                    # Average CI width
                    width = ci_df['upper_bound'] - ci_df['lower_bound']
                    metrics['average_width'][method] = width.mean()
                    
                    # CI asymmetry
                    forecast_center = ci_df['forecast']
                    upper_distance = ci_df['upper_bound'] - forecast_center
                    lower_distance = forecast_center - ci_df['lower_bound']
                    metrics['asymmetry'][method] = (
                        upper_distance.mean() - lower_distance.mean()
                    )
                    
                    # Relative width
                    metrics['relative_width'][method] = (
                        width.mean() / forecast_center.mean()
                    )
            
            return metrics
        except Exception as e:
            logger.error(f"Error in CI metrics calculation: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Generate sample historical data with trend and seasonality
    t = np.arange(len(dates))
    historical_data = pd.Series(
        100 + 0.1 * t + 20 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 10, len(t)),
        index=dates
    )
    
    # Generate sample forecast
    forecast_horizon = 30
    forecast_dates = pd.date_range(
        start=dates[-1] + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq='D'
    )
    forecast = pd.Series(
        historical_data.iloc[-1] + np.random.normal(0, 15, forecast_horizon),
        index=forecast_dates
    )
    
    # Calculate confidence intervals
    ci_calculator = ConfidenceIntervalCalculator(confidence_level=0.95)
    ci_results = ci_calculator.calculate_confidence_intervals(
        forecast,
        historical_data
    )
    
    # Print results
    print("\nConfidence Interval Metrics:")
    print(pd.DataFrame(ci_results['metrics']).round(2)) 