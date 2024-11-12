import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Receiving Overview", page_icon="📊", layout="wide")

def clean_bd_carousel_data(df):
    """Clean and restructure the BD Carousel receiving data"""
    # Skip the first 4 rows when reading
    df = df.iloc[4:].reset_index(drop=True)
    
    # Set the first row as header
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    
    # Select and rename only the columns we need
    columns_needed = {
        'Date / Time': 'Date',
        'Item ID': 'Med_ID',
        'Description': 'Description',
        'Quantity': 'Quantity'
    }
    
    # Select only needed columns
    df = df[list(columns_needed.keys())]
    
    # Rename columns
    df = df.rename(columns=columns_needed)
    
    # Remove empty rows
    df = df.dropna(how='all')
    
    # Clean up the data
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    
    return df

def get_top_medications(df, n=10):
    """Get top n unique medications, combining duplicates and keeping the description of the highest quantity"""
    # Create a copy of the dataframe to work with
    df_work = df.copy()
    
    # First, find the highest quantity entry for each Med_ID to get the description
    max_quantity_indices = df_work.groupby('Med_ID')['Quantity'].idxmax()
    descriptions = df_work.loc[max_quantity_indices, ['Med_ID', 'Description']]
    
    # Calculate total quantities for each Med_ID
    total_quantities = df_work.groupby('Med_ID')['Quantity'].sum().reset_index()
    
    # Merge descriptions with total quantities
    result = pd.merge(
        total_quantities,
        descriptions,
        on='Med_ID',
        how='left'
    )
    
    # Sort by total quantity
    result = result.sort_values('Quantity', ascending=False)
    
    # If we have less than n unique medications, print a warning
    if len(result) < n:
        st.warning(f"Only found {len(result)} unique medications, which is less than the requested {n}")
    
    # Get exactly n entries, padding with empty entries if necessary
    if len(result) < n:
        # Create empty rows to pad the result
        empty_rows = pd.DataFrame({
            'Med_ID': [f'Med_{i}' for i in range(len(result), n)],
            'Description': ['No Data' for _ in range(n - len(result))],
            'Quantity': [0 for _ in range(n - len(result))]
        })
        result = pd.concat([result, empty_rows], ignore_index=True)
    else:
        result = result.head(n)
    
    # Reset index for clean output
    return result.reset_index(drop=True)

def main():
    st.title("📊 Receiving Overview")
    
    # Add report type selection
    report_type = st.selectbox(
        "Select Report Type",
        options=["BD Carousel", "Omnicell"],
        index=0,
        help="Choose the type of receiving report you want to analyze"
    )
    
    if report_type == "BD Carousel":
        st.info("Upload BD Carousel receiving report to see top medications by quantity.")
    elif report_type == "Omnicell":
        st.info("Upload Omnicell receiving report. Make sure the data follows the Omnicell format.")
    
    uploaded_file = st.file_uploader(f"Upload {report_type} Data (Excel file)", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        if report_type == "BD Carousel":
            # Load and clean BD Carousel data
            df = pd.read_excel(uploaded_file)
            df_cleaned = clean_bd_carousel_data(df)
            
            if df_cleaned is not None:
                # Get top medications with combined duplicates
                top_meds = get_top_medications(df_cleaned, n=10)
                
                # Create display version with formatted numbers
                top_meds_display = top_meds.copy()
                top_meds_display['Quantity'] = top_meds_display['Quantity'].map('{:,.0f}'.format)
                
                # Display as a table
                st.subheader("Top 10 Medications by Quantity")
                st.dataframe(
                    top_meds_display,
                    column_config={
                        "Med_ID": "Medication ID",
                        "Description": "Medication Name",
                        "Quantity": "Total Quantity"
                    },
                    hide_index=True
                )
                
                # Create bar chart
                fig = px.bar(
                    top_meds,
                    x='Med_ID',
                    y='Quantity',
                    text='Quantity',
                    title='Top 10 Medications by Receiving Quantity',
                    hover_data=['Description'],
                    category_orders={"Med_ID": top_meds['Med_ID'].tolist()}
                )
                
                # Customize the chart
                fig.update_traces(
                    texttemplate='%{text:,.0f}',
                    textposition='outside',
                    marker_color='#1f77b4',
                    width=0.6
                )
                
                fig.update_layout(
                    height=500,
                    xaxis_title="Medication ID",
                    yaxis_title="Quantity",
                    showlegend=False,
                    xaxis={
                        'tickangle': 45,
                        'type': 'category',
                        'categoryorder': 'array',
                        'categoryarray': top_meds['Med_ID'].tolist()
                    },
                    bargap=0.2
                )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
        elif report_type == "Omnicell":
            st.info("Omnicell processing will be implemented based on the format")
    
    else:
        st.info(f"Please upload a {report_type} file to begin analysis.")

if __name__ == "__main__":
    main() 