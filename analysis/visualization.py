import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ForecastVisualizer:
    """
    A class for creating comprehensive visualizations of forecast results.
    """
    
    def __init__(self, style: str = 'seaborn'):
        """
        Initialize visualizer with style settings.
        
        Args:
            style: Visual style ('seaborn', 'plotly', or 'minimal')
        """
        self.style = style
        plt.style.use(style)
        self.colors = {
            'actual': '#2C3E50',
            'forecast': '#E74C3C',
            'confidence': '#3498DB',
            'trend': '#2ECC71',
            'seasonal': '#9B59B6',
            'residual': '#95A5A6'
        }
        logger.info(f"Initialized visualizer with {style} style")

    def create_comprehensive_dashboard(self,
                                    historical_data: pd.Series,
                                    forecast_results: Dict[str, Any],
                                    decomposition_results: Optional[Dict[str, pd.Series]] = None,
                                    output_path: Optional[str] = None) -> None:
        """
        Create a comprehensive dashboard of visualizations.
        """
        try:
            logger.info("Creating comprehensive forecast dashboard")
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 25))
            gs = fig.add_gridspec(5, 2)
            
            # 1. Main Forecast Plot
            ax1 = fig.add_subplot(gs[0, :])
            self._plot_forecast_with_ci(
                ax1, historical_data, forecast_results
            )
            
            # 2. Decomposition Plot
            if decomposition_results:
                ax2 = fig.add_subplot(gs[1, :])
                self._plot_decomposition(ax2, decomposition_results)
            
            # 3. Seasonal Patterns
            ax3 = fig.add_subplot(gs[2, 0])
            self._plot_seasonal_patterns(ax3, historical_data)
            
            # 4. Error Analysis
            ax4 = fig.add_subplot(gs[2, 1])
            self._plot_error_analysis(ax4, forecast_results)
            
            # 5. Trend Analysis
            ax5 = fig.add_subplot(gs[3, 0])
            self._plot_trend_analysis(ax5, historical_data)
            
            # 6. Forecast Distribution
            ax6 = fig.add_subplot(gs[3, 1])
            self._plot_forecast_distribution(ax6, forecast_results)
            
            # 7. Performance Metrics
            ax7 = fig.add_subplot(gs[4, :])
            self._plot_performance_metrics(ax7, forecast_results)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Dashboard saved to {output_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise

    def _plot_forecast_with_ci(self,
                             ax: plt.Axes,
                             historical_data: pd.Series,
                             forecast_results: Dict[str, Any]) -> None:
        """
        Plot forecast with confidence intervals.
        """
        try:
            # Plot historical data
            ax.plot(historical_data.index, historical_data,
                   color=self.colors['actual'],
                   label='Historical Data',
                   linewidth=2)
            
            # Plot forecast
            forecast = forecast_results['forecast']
            ci_lower = forecast_results['confidence_intervals']['lower_bound']
            ci_upper = forecast_results['confidence_intervals']['upper_bound']
            
            ax.plot(forecast.index, forecast,
                   color=self.colors['forecast'],
                   label='Forecast',
                   linewidth=2,
                   linestyle='--')
            
            # Plot confidence intervals
            ax.fill_between(forecast.index,
                          ci_lower,
                          ci_upper,
                          color=self.colors['confidence'],
                          alpha=0.2,
                          label=f'{int(forecast_results["confidence_level"]*100)}% CI')
            
            ax.set_title('Forecast with Confidence Intervals',
                        fontsize=14, pad=20)
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Error in forecast plot: {str(e)}")
            raise

    def _plot_decomposition(self,
                          ax: plt.Axes,
                          decomposition_results: Dict[str, pd.Series]) -> None:
        """
        Plot time series decomposition.
        """
        try:
            # Create three sub-axes
            gs = ax.get_gridspec()
            subax = gs.subgridspec(3, 1)
            
            # Plot trend
            ax1 = plt.subplot(subax[0])
            ax1.plot(decomposition_results['trend'],
                    color=self.colors['trend'],
                    label='Trend')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot seasonal
            ax2 = plt.subplot(subax[1])
            ax2.plot(decomposition_results['seasonal'],
                    color=self.colors['seasonal'],
                    label='Seasonal')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot residual
            ax3 = plt.subplot(subax[2])
            ax3.plot(decomposition_results['residual'],
                    color=self.colors['residual'],
                    label='Residual')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.suptitle('Time Series Decomposition',
                        fontsize=14, y=1.02)
            
        except Exception as e:
            logger.error(f"Error in decomposition plot: {str(e)}")
            raise

    def create_interactive_dashboard(self,
                                   historical_data: pd.Series,
                                   forecast_results: Dict[str, Any],
                                   decomposition_results: Optional[Dict[str, Any]] = None) -> go.Figure:
        """
        Create an interactive dashboard using Plotly.
        """
        try:
            logger.info("Creating interactive dashboard")
            
            # Create subplot figure
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Forecast with Confidence Intervals',
                    'Decomposition',
                    'Seasonal Patterns',
                    'Error Distribution',
                    'Trend Analysis',
                    'Performance Metrics'
                )
            )
            
            # Add forecast plot
            self._add_forecast_trace(fig, historical_data, forecast_results)
            
            # Add decomposition if available
            if decomposition_results:
                self._add_decomposition_traces(fig, decomposition_results)
            
            # Add other traces
            self._add_seasonal_trace(fig, historical_data)
            self._add_error_trace(fig, forecast_results)
            self._add_trend_trace(fig, historical_data)
            self._add_metrics_trace(fig, forecast_results)
            
            # Update layout
            fig.update_layout(
                height=1200,
                width=1600,
                showlegend=True,
                title_text="Forecast Analysis Dashboard",
                title_x=0.5,
                title_font_size=20
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {str(e)}")
            raise

    def _add_forecast_trace(self,
                          fig: go.Figure,
                          historical_data: pd.Series,
                          forecast_results: Dict[str, Any]) -> None:
        """
        Add forecast traces to interactive plot.
        """
        try:
            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=historical_data,
                    name="Historical",
                    line=dict(color=self.colors['actual'])
                ),
                row=1, col=1
            )
            
            # Forecast
            forecast = forecast_results['forecast']
            fig.add_trace(
                go.Scatter(
                    x=forecast.index,
                    y=forecast,
                    name="Forecast",
                    line=dict(color=self.colors['forecast'], dash='dash')
                ),
                row=1, col=1
            )
            
            # Confidence intervals
            ci_lower = forecast_results['confidence_intervals']['lower_bound']
            ci_upper = forecast_results['confidence_intervals']['upper_bound']
            
            fig.add_trace(
                go.Scatter(
                    x=forecast.index.tolist() + forecast.index.tolist()[::-1],
                    y=ci_upper.tolist() + ci_lower.tolist()[::-1],
                    fill='toself',
                    fillcolor=f"rgba{tuple(int(self.colors['confidence'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}",
                    line=dict(color='rgba(255,255,255,0)'),
                    name="Confidence Interval"
                ),
                row=1, col=1
            )
            
        except Exception as e:
            logger.error(f"Error adding forecast traces: {str(e)}")
            raise

    def export_visualizations(self,
                            output_dir: str,
                            formats: List[str] = ['png', 'html', 'pdf']) -> None:
        """
        Export visualizations in multiple formats.
        """
        try:
            logger.info(f"Exporting visualizations to {output_dir}")
            
            # Create static visualizations
            if 'png' in formats or 'pdf' in formats:
                self.create_comprehensive_dashboard(
                    output_path=f"{output_dir}/forecast_dashboard"
                )
            
            # Create interactive visualization
            if 'html' in formats:
                fig = self.create_interactive_dashboard()
                fig.write_html(f"{output_dir}/interactive_dashboard.html")
            
            logger.info("Successfully exported visualizations")
            
        except Exception as e:
            logger.error(f"Error exporting visualizations: {str(e)}")
            raise 