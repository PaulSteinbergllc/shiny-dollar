import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ABCAnalyzer:
    """
    A class for performing ABC (Pareto) analysis on inventory data.
    """
    
    def __init__(self, categories: Dict[str, float] = None):
        """
        Initialize ABC analyzer with category thresholds.
        
        Args:
            categories: Dictionary defining category thresholds
                       Default: A (0-80%), B (80-95%), C (95-100%)
        """
        self.categories = categories or {
            'A': 0.80,  # Top 80% of value
            'B': 0.95,  # Next 15% of value
            'C': 1.00   # Last 5% of value
        }
        logger.info("ABC Analyzer initialized with categories: %s", self.categories)

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform ABC analysis on the dataset.
        
        Args:
            df: DataFrame with 'NDC' and 'Total Value' columns
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info("Starting ABC analysis")
            
            # Validate input data
            self._validate_data(df)
            
            # Perform analysis
            abc_df = self._calculate_abc_metrics(df)
            summary = self._generate_summary(abc_df)
            
            # Create visualizations
            plots = self._create_visualizations(abc_df)
            
            results = {
                'abc_classification': abc_df,
                'summary': summary,
                'plots': plots
            }
            
            logger.info("ABC analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error("Error in ABC analysis: %s", str(e))
            raise

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate input data."""
        required_columns = {'NDC', 'Total Value'}
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        if df['Total Value'].isnull().any():
            raise ValueError("Total Value column contains null values")
            
        logger.info("Data validation passed")

    def _calculate_abc_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ABC analysis metrics."""
        try:
            # Group by NDC and sum values
            abc_df = df.groupby('NDC')['Total Value'].sum().reset_index()
            
            # Sort by value in descending order
            abc_df = abc_df.sort_values('Total Value', ascending=False)
            
            # Calculate percentages
            total_value = abc_df['Total Value'].sum()
            abc_df['Value Percentage'] = abc_df['Total Value'] / total_value * 100
            abc_df['Cumulative Percentage'] = abc_df['Value Percentage'].cumsum()
            
            # Assign categories
            abc_df['Category'] = abc_df['Cumulative Percentage'].apply(self._assign_category)
            
            # Calculate rankings
            abc_df['Rank'] = range(1, len(abc_df) + 1)
            
            return abc_df
            
        except Exception as e:
            logger.error("Error in metric calculation: %s", str(e))
            raise

    def _assign_category(self, cumulative_percentage: float) -> str:
        """Assign ABC category based on cumulative percentage."""
        for category, threshold in self.categories.items():
            if cumulative_percentage <= threshold * 100:
                return category
        return 'C'  # Default category

    def _generate_summary(self, abc_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Generate summary statistics for each category."""
        summary = {}
        
        for category in ['A', 'B', 'C']:
            cat_data = abc_df[abc_df['Category'] == category]
            summary[category] = {
                'item_count': len(cat_data),
                'total_value': cat_data['Total Value'].sum(),
                'percentage_of_items': len(cat_data) / len(abc_df) * 100,
                'percentage_of_value': cat_data['Value Percentage'].sum()
            }
            
        return summary

    def _create_visualizations(self, abc_df: pd.DataFrame) -> Dict[str, plt.Figure]:
        """Create visualization plots."""
        plots = {}
        
        # Pareto Chart
        fig_pareto = plt.figure(figsize=(12, 6))
        ax1 = fig_pareto.add_subplot(111)
        ax2 = ax1.twinx()
        
        # Plot bars
        ax1.bar(range(len(abc_df)), abc_df['Value Percentage'], 
                color=abc_df['Category'].map({'A': 'green', 'B': 'yellow', 'C': 'red'}))
        
        # Plot cumulative line
        ax2.plot(range(len(abc_df)), abc_df['Cumulative Percentage'], 
                color='black', linewidth=2)
        
        # Customize plot
        ax1.set_xlabel('NDC Rank')
        ax1.set_ylabel('Value Percentage')
        ax2.set_ylabel('Cumulative Percentage')
        plt.title('ABC Analysis - Pareto Chart')
        
        plots['pareto'] = fig_pareto
        
        # Category Distribution
        fig_dist = plt.figure(figsize=(10, 6))
        summary = self._generate_summary(abc_df)
        
        categories = []
        item_percentages = []
        value_percentages = []
        
        for category in ['A', 'B', 'C']:
            categories.append(category)
            item_percentages.append(summary[category]['percentage_of_items'])
            value_percentages.append(summary[category]['percentage_of_value'])
        
        x = range(len(categories))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], item_percentages, width, 
                label='% of Items', color='lightblue')
        plt.bar([i + width/2 for i in x], value_percentages, width, 
                label='% of Value', color='lightgreen')
        
        plt.xlabel('Category')
        plt.ylabel('Percentage')
        plt.title('ABC Analysis - Category Distribution')
        plt.xticks(x, categories)
        plt.legend()
        
        plots['distribution'] = fig_dist
        
        return plots

def print_summary(summary: Dict[str, Dict[str, float]]) -> None:
    """Print formatted summary of ABC analysis."""
    print("\nABC Analysis Summary:")
    print("-" * 80)
    print(f"{'Category':<10} {'Items':<10} {'% of Items':<12} {'Total Value':<15} {'% of Value':<12}")
    print("-" * 80)
    
    for category, metrics in summary.items():
        print(f"{category:<10} "
              f"{metrics['item_count']:<10} "
              f"{metrics['percentage_of_items']:,.1f}% "
              f"${metrics['total_value']:,.2f} "
              f"{metrics['percentage_of_value']:,.1f}%")
    print("-" * 80)

# Example usage
if __name__ == "__main__":
    # Sample data or load your Excel file
    df = pd.read_excel("C:/Users/super/OneDrive/Desktop/Cursor/Carousel/Cardinal1.xlsx")
    
    # Initialize analyzer
    analyzer = ABCAnalyzer()
    
    # Perform analysis
    results = analyzer.analyze(df)
    
    # Print summary
    print_summary(results['summary'])
    
    # Show plots
    plt.show()
