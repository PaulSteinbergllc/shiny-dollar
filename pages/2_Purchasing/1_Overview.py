import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configure page settings
st.set_page_config(
    page_title="Overview",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS for a bright theme
st.markdown("""
    <style>
        .stApp {
            background-color: #f8f9fa;
        }
        
        .css-1kyxreq, .css-12oz5g7 {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        [data-testid="stMetricValue"] {
            color: #2c3e50;
            font-size: 24px;
        }
        
        h1, h2, h3 {
            color: #2c3e50;
        }
        
        .stDataFrame {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

def format_currency(value):
    return f"${value:,.2f}"

def main():
    st.empty()
    
    st.title("📊 Overview")
    
    if 'processed_df' not in st.session_state or not st.session_state['analysis_run']:
        st.error("No data available. Please run the analysis from the Home page first.")
        return
    
    df = st.session_state['processed_df']
    
    if df is not None:
        # Calculate summary statistics
        total_price = df['Price'].sum()
        total_quantity = df['Quantity'].sum()
        unique_ndcs = df['NDC'].nunique()
        
        # Calculate date range
        start_date = df['Date'].min().strftime('%m/%Y')
        end_date = df['Date'].max().strftime('%m/%Y')
        
        # Display metrics with duration first
        col1, col2, col3, col4 = st.columns([1,1,1,1])
        with col1:
            st.markdown(f"### Duration: {start_date} - {end_date}")
        with col2:
            st.markdown(f"### Total NDCs: {unique_ndcs:,}")
        with col3:
            st.markdown(f"### Total Expenditure: ${total_price:,.2f}")
        with col4:
            st.markdown(f"### Total Quantity: {total_quantity:,} units")
        
        # Monthly expenditure analysis
        st.subheader("Monthly Expenditure")
        
        # Calculate monthly totals
        monthly_data = df.groupby(df['Date'].dt.strftime('%b-%Y'))['Price'].sum().reset_index()
        monthly_data['Date'] = pd.to_datetime(monthly_data['Date'], format='%b-%Y')
        monthly_data = monthly_data.sort_values('Date')
        
        # Create the line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_data['Date'].dt.strftime('%b-%Y'),
            y=monthly_data['Price'],
            mode='lines+markers',
            name='Monthly Expenditure',
            line=dict(color='#00BFB3', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            xaxis=dict(
                title='Month',
                titlefont=dict(size=16, color='black'),
                tickfont=dict(size=14, color='black'),
                tickangle=45,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            yaxis=dict(
                title='Total Expenditure ($)',
                titlefont=dict(size=16, color='black'),
                tickfont=dict(size=14, color='black'),
                tickformat='$,.0f',
                gridcolor='rgba(0,0,0,0.1)'
            ),
            showlegend=False,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=50, l=0, r=0, b=0),
            height=600
        )
        
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>$%{y:,.2f}<extra></extra>'
        )
        
        container = st.container()
        with container:
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()


