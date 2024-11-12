# Import statements and TrendAnalyzer class from previous code
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrendAnalyzer:
    """
    A class for comprehensive trend analysis of time series data.
    
    Attributes:
        data (pd.Series): Time series data indexed by date
        decomposition: Seasonal decomposition results
        trends (dict): Storage for computed trends
    """
    
    def __init__(self, data: pd.Series):
        """
        Initialize trend analyzer with time series data.
        
        Args:
            data: Time series data indexed by date
        """
        self._validate_input(data)
        self.data = data
        self.decomposition = None
        self.trends = {}
        logger.info("TrendAnalyzer initialized successfully")

    def _validate_input(self, data: pd.Series) -> None:
        """
        Validate input data.
        
        Args:
            data: Input time series data
            
        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Data must be a pandas Series")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Series must be indexed by dates")
        if data.isnull().any():
            raise ValueError("Series contains null values")
        logger.info("Input data validated successfully")

    def analyze_all_trends(self) -> Dict[str, Any]:
        """
        Perform comprehensive trend analysis.
        
        Returns:
            Dictionary containing all trend analysis results
        """
        try:
            logger.info("Starting comprehensive trend analysis")
            results = {
                'decomposition': self._decompose_series(),
                'short_term_trends': self._analyze_short_term_trends(),
                'long_term_trends': self._analyze_long_term_trends(),
                'seasonal_patterns': self._analyze_seasonal_patterns(),
                'trend_statistics': self._calculate_trend_statistics()
            }
            logger.info("Completed comprehensive trend analysis")
            return results
        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            raise

    def _decompose_series(self) -> Dict[str, pd.Series]:
        """
        Decompose time series into trend, seasonal, and residual components.
        """
        try:
            logger.info("Starting series decomposition")
            self.decomposition = seasonal_decompose(
                self.data,
                period=self._estimate_seasonal_period(),
                extrapolate_trend='freq'
            )
            
            result = {
                'trend': self.decomposition.trend,
                'seasonal': self.decomposition.seasonal,
                'residual': self.decomposition.resid
            }
            logger.info("Completed series decomposition")
            return result
        except Exception as e:
            logger.error(f"Error in series decomposition: {str(e)}")
            raise

    def _estimate_seasonal_period(self) -> int:
        """
        Estimate the seasonal period from the data.
        """
        # Check for common periods (daily, weekly, monthly)
        frequencies = {
            7: 'weekly',
            30: 'monthly',
            365: 'yearly'
        }
        
        best_period = 7  # Default to weekly
        max_acf = 0
        
        for period in frequencies.keys():
            if len(self.data) >= period * 2:
                acf = self.data.autocorr(lag=period)
                if acf > max_acf:
                    max_acf = acf
                    best_period = period
        
        logger.info(f"Estimated seasonal period: {best_period}")
        return best_period

    def _analyze_short_term_trends(self) -> Dict[str, Any]:
        """
        Analyze short-term trends using rolling statistics.
        """
        try:
            logger.info("Analyzing short-term trends")
            
            # Calculate rolling statistics
            rolling_7d = self.data.rolling(window=7)
            rolling_30d = self.data.rolling(window=30)
            
            # Calculate momentum indicators
            roc_7d = self.data.pct_change(periods=7)
            roc_30d = self.data.pct_change(periods=30)
            
            # Detect trend changes
            trend_changes = self._detect_trend_changes(rolling_30d.mean())
            
            result = {
                'rolling_stats': {
                    'weekly_mean': rolling_7d.mean(),
                    'monthly_mean': rolling_30d.mean(),
                    'weekly_std': rolling_7d.std(),
                    'monthly_std': rolling_30d.std()
                },
                'momentum': {
                    'weekly_roc': roc_7d,
                    'monthly_roc': roc_30d
                },
                'trend_changes': trend_changes,
                'volatility': self._calculate_volatility()
            }
            
            logger.info("Completed short-term trend analysis")
            return result
        except Exception as e:
            logger.error(f"Error in short-term trend analysis: {str(e)}")
            raise

    def _analyze_long_term_trends(self) -> Dict[str, Any]:
        """
        Analyze long-term trends using regression and statistical tests.
        """
        try:
            logger.info("Analyzing long-term trends")
            
            # Prepare data for regression
            X = np.arange(len(self.data)).reshape(-1, 1)
            y = self.data.values
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Perform Mann-Kendall test
            mk_result = self._mann_kendall_test(self.data)
            
            # Perform Augmented Dickey-Fuller test
            adf_result = adfuller(self.data)
            
            result = {
                'regression': {
                    'slope': model.coef_[0],
                    'intercept': model.intercept_,
                    'trend_direction': 'increasing' if model.coef_[0] > 0 else 'decreasing',
                    'r_squared': model.score(X, y)
                },
                'mann_kendall': mk_result,
                'stationarity': {
                    'adf_statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'is_stationary': adf_result[1] < 0.05
                }
            }
            
            logger.info("Completed long-term trend analysis")
            return result
        except Exception as e:
            logger.error(f"Error in long-term trend analysis: {str(e)}")
            raise

    def _analyze_seasonal_patterns(self) -> Dict[str, Any]:
        """
        Analyze seasonal patterns at different frequencies.
        """
        try:
            logger.info("Analyzing seasonal patterns")
            
            # Calculate patterns at different frequencies
            patterns = {
                'daily': self.data.groupby(self.data.index.hour).mean()
                        if self.data.index.hour.nunique() > 1 else None,
                'weekly': self.data.groupby(self.data.index.dayofweek).mean(),
                'monthly': self.data.groupby(self.data.index.month).mean()
            }
            
            # Calculate seasonal strength
            seasonal_strength = self._calculate_seasonal_strength()
            
            # Identify peak periods
            peaks = self._identify_peak_periods()
            
            result = {
                'patterns': patterns,
                'seasonal_strength': seasonal_strength,
                'peak_periods': peaks
            }
            
            logger.info("Completed seasonal pattern analysis")
            return result
        except Exception as e:
            logger.error(f"Error in seasonal pattern analysis: {str(e)}")
            raise

    def _calculate_trend_statistics(self) -> Dict[str, float]:
        """
        Calculate various trend statistics.
        """
        try:
            logger.info("Calculating trend statistics")
            
            result = {
                'trend_strength': self._calculate_trend_strength(),
                'volatility': self.data.std() / self.data.mean(),
                'trend_reliability': self._calculate_trend_reliability()
            }
            
            logger.info("Completed trend statistics calculation")
            return result
        except Exception as e:
            logger.error(f"Error in trend statistics calculation: {str(e)}")
            raise

    def _detect_trend_changes(self, rolling_mean: pd.Series) -> pd.Series:
        """
        Detect significant changes in trend direction.
        """
        diff = rolling_mean.diff()
        sign_changes = ((diff.shift(1) * diff) < 0) & (diff.abs() > diff.std())
        return sign_changes[sign_changes].index

    def _mann_kendall_test(self, data: pd.Series) -> Dict[str, Any]:
        """
        Perform Mann-Kendall trend test.
        """
        n = len(data)
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(data.iloc[j] - data.iloc[i])
        
        variance = (n * (n - 1) * (2 * n + 5)) / 18
        if s > 0:
            z = (s - 1) / np.sqrt(variance)
        elif s < 0:
            z = (s + 1) / np.sqrt(variance)
        else:
            z = 0
            
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {
            'statistic': z,
            'p_value': p_value,
            'trend': 'increasing' if z > 0 else 'decreasing' if z < 0 else 'no trend'
        }

    def _calculate_seasonal_strength(self) -> float:
        """
        Calculate the strength of seasonality.
        """
        if self.decomposition is None:
            self._decompose_series()
        
        variance_seasonal = np.var(self.decomposition.seasonal)
        variance_residual = np.var(self.decomposition.resid)
        
        return variance_seasonal / (variance_seasonal + variance_residual)

    def _calculate_volatility(self) -> float:
        """
        Calculate the volatility of the time series.
        """
        return self.data.std() / self.data.mean()

    def _identify_peak_periods(self) -> Dict[str, Any]:
        """
        Identify peak periods at different frequencies.
        """
        return {
            'daily_peak': self.data.groupby(self.data.index.hour).mean().idxmax()
                         if self.data.index.hour.nunique() > 1 else None,
            'weekly_peak': self.data.groupby(self.data.index.dayofweek).mean().idxmax(),
            'monthly_peak': self.data.groupby(self.data.index.month).mean().idxmax()
        }

    def plot_comprehensive_analysis(self) -> None:
        """
        Create comprehensive visualization of trend analysis.
        """
        try:
            logger.info("Creating comprehensive visualization")
            
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(4, 2)
            
            # Original data with trend
            ax1 = fig.add_subplot(gs[0, :])
            self._plot_data_with_trend(ax1)
            
            # Decomposition
            ax2 = fig.add_subplot(gs[1, 0])
            self._plot_decomposition(ax2)
            
            # Seasonal patterns
            ax3 = fig.add_subplot(gs[1, 1])
            self._plot_seasonal_patterns(ax3)
            
            # Short-term trends
            ax4 = fig.add_subplot(gs[2, 0])
            self._plot_short_term_trends(ax4)
            
            # Long-term trends
            ax5 = fig.add_subplot(gs[2, 1])
            self._plot_long_term_trends(ax5)
            
            # Statistics summary
            ax6 = fig.add_subplot(gs[3, :])
            self._plot_statistics_summary(ax6)
            
            plt.tight_layout()
            plt.show()
            
            logger.info("Completed visualization")
        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")
            raise

    def _plot_data_with_trend(self, ax):
        """Plot original data with trend line."""
        ax.plot(self.data.index, self.data, label='Original Data')
        if self.decomposition is not None:
            ax.plot(self.data.index, self.decomposition.trend, 
                   label='Trend', color='red')
        ax.set_title('Time Series with Trend')
        ax.legend()
        ax.grid(True)

    def _plot_decomposition(self, ax):
        """Plot decomposition components."""
        if self.decomposition is not None:
            ax.plot(self.data.index, self.decomposition.seasonal, 
                   label='Seasonal')
            ax.plot(self.data.index, self.decomposition.resid, 
                   label='Residual')
        ax.set_title('Decomposition Components')
        ax.legend()
        ax.grid(True)

    def _plot_seasonal_patterns(self, ax):
        """Plot seasonal patterns."""
        if self.decomposition is not None:
            seasonal = self.decomposition.seasonal
            ax.boxplot([seasonal[seasonal.index.dayofweek == i] 
                       for i in range(7)])
            ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax.set_title('Weekly Seasonal Patterns')
        ax.grid(True)

    def _plot_short_term_trends(self, ax):
        """Plot short-term trends."""
        rolling_mean = self.data.rolling(window=7).mean()
        rolling_std = self.data.rolling(window=7).std()
        ax.plot(self.data.index, rolling_mean, label='7-day Moving Average')
        ax.fill_between(self.data.index, 
                       rolling_mean - rolling_std,
                       rolling_mean + rolling_std,
                       alpha=0.2)
        ax.set_title('Short-term Trends')
        ax.legend()
        ax.grid(True)

    def _plot_long_term_trends(self, ax):
        """Plot long-term trends."""
        X = np.arange(len(self.data)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, self.data.values)
        trend_line = model.predict(X)
        ax.plot(self.data.index, trend_line, label='Long-term Trend',
                color='red')
        ax.plot(self.data.index, self.data, label='Original Data',
                alpha=0.5)
        ax.set_title('Long-term Trend')
        ax.legend()
        ax.grid(True)

    def _plot_statistics_summary(self, ax):
        """Plot statistics summary."""
        stats = self._calculate_trend_statistics()
        summary = (
            f"Trend Strength: {stats['trend_strength']:.2f}\n"
            f"Volatility: {stats['volatility']:.2f}\n"
            f"Trend Reliability: {stats['trend_reliability']:.2f}\n"
        )
        ax.text(0.5, 0.5, summary, ha='center', va='center', wrap=True)