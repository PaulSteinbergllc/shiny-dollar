import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import locale

# Set locale for currency formatting
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'en_US')
    except:
        pass  # Fallback to default locale if neither works

# Initialize session state for data persistence
if 'processed_df' not in st.session_state:
    st.session_state['processed_df'] = None
if 'analysis_run' not in st.session_state:
    st.session_state['analysis_run'] = False
if 'data_info' not in st.session_state:
    st.session_state['data_info'] = {}

# Configure page settings
st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)

def create_column_mapping(df):
    # Required column names
    required_columns = ['NDC', 'Date', 'Quantity', 'Price']
    
    # Create mapping for each required column
    col_mapping = {}
    
    # Create columns for mapping selection with original width
    for req_col in required_columns:
        col_mapping[req_col] = st.selectbox(
            f"{req_col} *",
            options=df.columns,
            key=f"map_{req_col}",
            # Add these parameters to maintain original width
            label_visibility="visible",
            help=None,
            disabled=False,
            kwargs={"style": "width: 200px;"}  # Set specific width
        )
    
    return col_mapping

def format_currency(value):
    try:
        return locale.currency(float(value), grouping=True)
    except:
        return value

def format_ndc(ndc):
    """Format NDC to 5-4-2 format without decimal"""
    try:
        # Remove any non-numeric characters and .0
        ndc = str(ndc).replace('.0', '')
        ndc = ''.join(filter(str.isdigit, ndc))
        
        # Format to 5-4-2
        if len(ndc) == 11:
            return f"{ndc[:5]}-{ndc[5:9]}-{ndc[9:]}"
        return ndc
    except:
        return ndc

def process_file(uploaded_file):
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)
        
        # Create a preview dataframe with formatted NDC
        preview_df = df.copy()
        ndc_col = preview_df.columns[0]  # Assuming NDC is the first column, will be updated after mapping
        preview_df[ndc_col] = preview_df[ndc_col].apply(format_ndc)
        
        # Display raw data preview
        st.subheader("Raw Data Preview")
        st.dataframe(preview_df.head())
        
        # Get column mappings from user
        st.subheader("Column Mapping")
        st.write("Map your columns to the required fields:")
        
        col_mapping = create_column_mapping(df)
        
        # Show mapped preview if all columns are selected
        if len(set(col_mapping.values())) == 4:  # All 4 columns are mapped
            mapped_preview = df.copy()
            # Rename columns based on mapping
            mapped_preview = mapped_preview.rename(columns={
                col_mapping['NDC']: 'NDC',
                col_mapping['Date']: 'Date',
                col_mapping['Quantity']: 'Quantity',
                col_mapping['Price']: 'Price'
            })
            # Format NDC in the preview
            mapped_preview['NDC'] = mapped_preview['NDC'].apply(format_ndc)
            
            st.subheader("Mapped Data Preview")
            st.dataframe(mapped_preview[['NDC', 'Date', 'Quantity', 'Price']].head())
        
        if st.button("Run Analysis", type="primary"):
            if col_mapping:
                # Create processed dataframe
                processed_df = df.copy()
                
                # Rename columns based on mapping
                processed_df = processed_df.rename(columns={
                    col_mapping['NDC']: 'NDC',
                    col_mapping['Date']: 'Date',
                    col_mapping['Quantity']: 'Quantity',
                    col_mapping['Price']: 'Price'
                })
                
                try:
                    # Convert date
                    processed_df['Date'] = pd.to_datetime(processed_df['Date'])
                    
                    # Clean and convert price
                    def clean_price(price):
                        if pd.isna(price):
                            return 0.0
                        if isinstance(price, str):
                            price = price.replace('$', '').replace(',', '')
                            if '(' in price and ')' in price:
                                price = '-' + price.replace('(', '').replace(')', '')
                        return float(price)
                    
                    processed_df['Price'] = processed_df['Price'].apply(clean_price)
                    
                    # Convert quantity with NaN handling
                    def clean_quantity(qty):
                        if pd.isna(qty):
                            return 0
                        if isinstance(qty, str):
                            qty = qty.replace(',', '')
                            if '(' in qty and ')' in qty:
                                qty = '-' + qty.replace('(', '').replace(')', '')
                        return int(float(qty))
                    
                    processed_df['Quantity'] = processed_df['Quantity'].apply(clean_quantity)
                    
                    # Remove rows with zero quantity or price
                    processed_df = processed_df[
                        (processed_df['Quantity'] != 0) & 
                        (processed_df['Price'] != 0)
                    ]
                    
                    # Format NDC
                    processed_df['NDC'] = processed_df['NDC'].astype(str).str.replace('.0', '')
                    processed_df['NDC'] = processed_df['NDC'].apply(format_ndc)
                    
                    # Store in session state
                    st.session_state['processed_df'] = processed_df
                    st.session_state['analysis_run'] = True
                    
                    # Navigate to Overview
                    st.switch_page("pages/1_Overview.py")
                    
                except Exception as e:
                    st.error(f"Error processing columns: {str(e)}")
                    return None
            else:
                st.error("Please map all required columns before running the analysis.")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def main():
    st.title("📊 Purchasing Data")
    
    # Add tabs for navigation
    tab1, tab2, tab3 = st.tabs(["Main", "Overview", "Analysis"])
    
    with tab1:
        # Your main purchasing page content
        pass
        
    with tab2:
        if st.button("Go to Overview"):
            st.switch_page("pages/1_Purchasing/1_Overview.py")
            
    with tab3:
        if st.button("Go to Analysis"):
            st.switch_page("pages/1_Purchasing/2_Analysis.py")

    # Show data info if available
    if st.session_state['analysis_run'] and st.session_state['data_info']:
        info = st.session_state['data_info']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{info['total_records']:,}")
        with col2:
            st.metric("Unique Products", f"{info['unique_products']:,}")
        with col3:
            st.metric("Total Spend", info['total_spend'])
        st.info(f"Date Range: {info['date_range']}")
    
    uploaded_file = st.file_uploader("Drag and drop file here", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        process_file(uploaded_file)

if __name__ == "__main__":
    main() 