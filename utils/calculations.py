import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import streamlit as st

class AnalysisCalculator:
    @staticmethod
    def calculate_abc_analysis(data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate ABC analysis"""
        try:
            # Calculate total value for each NDC
            ndc_values = data.groupby('NDC').apply(
                lambda x: (x['Quantity'] * x['Price']).sum()
            ).sort_values(ascending=False)
            
            total_value = ndc_values.sum()
            cumulative_percentage = (ndc_values.cumsum() / total_value) * 100
            
            # Categorize items
            categories = pd.cut(
                cumulative_percentage,
                bins=[0, 80, 95, 100],
                labels=['A', 'B', 'C']
            )
            
            results = {
                'A': {'count': 0, 'value': 0, 'percentage': 0},
                'B': {'count': 0, 'value': 0, 'percentage': 0},
                'C': {'count': 0, 'value': 0, 'percentage': 0}
            }
            
            for cat in ['A', 'B', 'C']:
                mask = categories == cat
                results[cat]['count'] = mask.sum()
                results[cat]['value'] = ndc_values[mask].sum()
                results[cat]['percentage'] = (results[cat]['value'] / total_value) * 100
                
            return results
        except Exception as e:
            st.error(f"Error in ABC analysis: {str(e)}")
            return None

    @staticmethod
    def calculate_trends(data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trends from the data"""
        try:
            # Monthly aggregation
            monthly_data = data.groupby(data['Date'].dt.to_period('M')).agg({
                'Quantity': 'sum',
                'Price': 'sum',
                'NDC': 'nunique'
            }).reset_index()
            
            # Calculate month-over-month changes
            trends = {}
            for col in ['Quantity', 'Price', 'NDC']:
                pct_change = monthly_data[col].pct_change().iloc[-1]
                trends[f'Monthly {col} Change'] = f"{'↑' if pct_change > 0 else '↓'} {abs(pct_change):.1%}"
            
            # Top NDCs
            top_ndcs = data.groupby('NDC').agg({
                'Quantity': 'sum',
                'Price': 'sum'
            }).sort_values('Quantity', ascending=False).head(5)
            
            trends['Top NDCs'] = top_ndcs
            
            return trends
        except Exception as e:
            st.error(f"Error in trend analysis: {str(e)}")
            return None

    @staticmethod
    def detect_anomalies(data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in the data"""
        try:
            anomalies = {}
            
            # Quantity anomalies (using z-score method)
            z_scores = np.abs((data['Quantity'] - data['Quantity'].mean()) / data['Quantity'].std())
            quantity_anomalies = data[z_scores > 3][['Date', 'NDC', 'Quantity']]
            
            # Price anomalies
            z_scores = np.abs((data['Price'] - data['Price'].mean()) / data['Price'].std())
            price_anomalies = data[z_scores > 3][['Date', 'NDC', 'Price']]
            
            # Sudden changes
            daily_data = data.groupby('Date').agg({
                'Quantity': 'sum',
                'Price': 'sum'
            })
            
            pct_changes = daily_data.pct_change()
            spikes = {
                'Quantity': len(pct_changes[abs(pct_changes['Quantity']) > 0.5]),
                'Price': len(pct_changes[abs(pct_changes['Price']) > 0.5])
            }
            
            return {
                'Quantity': {
                    'count': len(quantity_anomalies),
                    'details': quantity_anomalies.head(3)
                },
                'Price': {
                    'count': len(price_anomalies),
                    'details': price_anomalies.head(3)
                },
                'Spikes': spikes
            }
        except Exception as e:
            st.error(f"Error in anomaly detection: {str(e)}")
            return None