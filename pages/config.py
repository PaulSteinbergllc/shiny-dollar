import streamlit as st
import pandas as pd

def setup_page():
    # Configure page settings
    st.set_page_config(
        page_title="Inventory Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    # Add custom CSS for styling
    st.markdown("""
        <style>
            /* Main background */
            .stApp {
                background-color: #f8f9fa;
            }
            
            /* Sidebar */
            [data-testid="stSidebar"] {
                background-color: #ffffff;
            }
            
            /* Cards and containers */
            .css-1kyxreq, .css-12oz5g7 {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            
            /* Metrics */
            [data-testid="stMetricValue"] {
                color: #2c3e50;
                font-size: 24px;
            }
            
            /* Headers */
            h1, h2, h3 {
                color: #2c3e50;
            }
            
            /* Dataframe */
            .stDataFrame {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 10px;
            }
            
            /* Buttons */
            .stButton>button {
                background-color: #00BFB3;
                color: white;
                border-radius: 5px;
            }
            
            /* Select boxes */
            .stSelectbox {
                background-color: #ffffff;
            }
            
            /* Tabs styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 2px;
            }
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                background-color: #F0F2F6;
                border-radius: 4px 4px 0px 0px;
                gap: 1px;
                padding-top: 10px;
                padding-bottom: 10px;
            }
            .stTabs [aria-selected="true"] {
                background-color: #FFFFFF;
            }
            
            /* Container padding */
            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
            }
        </style>
    """, unsafe_allow_html=True)

# Shared utility functions
def format_ndc(ndc):
    """Format NDC into 5-4-2 format"""
    ndc = str(ndc).replace('-', '')
    ndc = ndc.zfill(11)
    return f"{ndc[:5]}-{ndc[5:9]}-{ndc[9:]}"

def format_currency(amount):
    """Format currency with $ and commas"""
    return f"${amount:,.2f}"

def format_number(number):
    """Format numbers with commas"""
    return f"{number:,}"

# Common data processing functions
def process_date_column(df, date_column='Date'):
    """Standardize date column processing"""
    df[date_column] = pd.to_datetime(df[date_column])
    return df

def process_ndc_column(df, ndc_column='NDC'):
    """Standardize NDC column processing"""
    df[ndc_column] = df[ndc_column].astype(str).str.zfill(11)
    return df

# Chart settings
CHART_THEME = {
    'bgcolor': '#FFFFFF',
    'font_family': 'Arial',
    'title_font_size': 24,
    'axis_font_size': 12,
    'primary_color': '#00BFB3',  # Matching button color
    'secondary_color': '#2c3e50'  # Matching header color
}

def apply_chart_style(fig):
    """Apply consistent styling to plotly figures"""
    fig.update_layout(
        plot_bgcolor=CHART_THEME['bgcolor'],
        paper_bgcolor=CHART_THEME['bgcolor'],
        font_family=CHART_THEME['font_family'],
        title_font_size=CHART_THEME['title_font_size'],
        font_size=CHART_THEME['axis_font_size']
    )
    return fig

# Error messages
ERROR_MESSAGES = {
    'no_file': "Please upload a file to begin analysis.",
    'invalid_format': "Invalid file format. Please upload an Excel file.",
    'missing_columns': "Required columns are missing in the file.",
    'data_processing': "Error processing the data. Please check the file format.",
}

# Required columns for each type of data
REQUIRED_COLUMNS = {
    'purchasing': ['Date', 'NDC', 'Quantity', 'Price'],
    'receiving': ['Date', 'NDC', 'Quantity', 'Price'],
    'waste': ['Date', 'NDC', 'Quantity', 'Reason']
}

# Initialize session state variables
def init_session_state():
    if 'analysis_run' not in st.session_state:
        st.session_state.analysis_run = False
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None