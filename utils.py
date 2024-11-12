import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import streamlit as st

class DataProcessor:
    def __init__(self):
        self.df = None
        
    def load_data(self, file):
        """Load and process the Excel file"""
        try:
            self.df = pd.read_excel(file)
            # Convert date columns to datetime
            date_columns = ['NDC Creation Date', 'NDC Expiration Date']
            for col in date_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col])
            return True
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False

    def get_basic_stats(self):
        """Calculate basic statistics"""
        if self.df is None:
            return None
        
        stats = {
            'total_ndcs': len(self.df),
            'total_value': self.df['Total Value'].sum() if 'Total Value' in self.df.columns else 0,
            'date_range': {
                'start': self.df['NDC Creation Date'].min() if 'NDC Creation Date' in self.df.columns else None,
                'end': self.df['NDC Expiration Date'].max() if 'NDC Expiration Date' in self.df.columns else None
            }
        }
        return stats

    def get_monthly_trends(self):
        """Calculate monthly trends"""
        if self.df is None:
            return None
        
        if 'NDC Creation Date' in self.df.columns:
            monthly_data = self.df.groupby(self.df['NDC Creation Date'].dt.to_period('M')).agg({
                'Total Value': 'sum',
                'NDC': 'count'
            }).reset_index()
            monthly_data['NDC Creation Date'] = monthly_data['NDC Creation Date'].astype(str)
            return monthly_data
        return None

class Visualizer:
    @staticmethod
    def create_monthly_trend_chart(monthly_data):
        """Create monthly trend visualization"""
        if monthly_data is None:
            return None
        
        fig = go.Figure()
        
        # Add Total Value line
        fig.add_trace(go.Scatter(
            x=monthly_data['NDC Creation Date'],
            y=monthly_data['Total Value'],
            name='Total Value',
            line=dict(color='blue'),
            yaxis='y'
        ))
        
        # Add NDC count line
        fig.add_trace(go.Scatter(
            x=monthly_data['NDC Creation Date'],
            y=monthly_data['NDC'],
            name='NDC Count',
            line=dict(color='red'),
            yaxis='y2'
        ))
        
        # Update layout
        fig.update_layout(
            title='Monthly Trends: Total Value and NDC Count',
            xaxis_title='Month',
            yaxis_title='Total Value ($)',
            yaxis2=dict(
                title='NDC Count',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified'
        )
        
        return fig

    @staticmethod
    def create_value_distribution_chart(df):
        """Create value distribution visualization"""
        if df is None:
            return None
        
        fig = px.histogram(
            df,
            x='Total Value',
            nbins=50,
            title='Distribution of Total Values'
        )
        fig.update_layout(
            xaxis_title='Total Value ($)',
            yaxis_title='Count'
        )
        return fig

class SessionState:
    """Manage session state across pages"""
    @staticmethod
    def get_data_processor():
        """Get or create DataProcessor instance"""
        if 'data_processor' not in st.session_state:
            st.session_state.data_processor = DataProcessor()
        return st.session_state.data_processor

    @staticmethod
    def get_visualizer():
        """Get or create Visualizer instance"""
        if 'visualizer' not in st.session_state:
            st.session_state.visualizer = Visualizer()
        return st.session_state.visualizer

    @staticmethod
    def clear():
        """Clear session state"""
        st.session_state.clear()

def format_currency(value):
    """Format number as currency"""
    return f"${value:,.2f}"

def format_date(date):
    """Format datetime as string"""
    if pd.isna(date):
        return "N/A"
    return date.strftime("%b %Y") 