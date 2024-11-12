import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="Receiving",
    page_icon="📦",
    layout="wide"
)

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

def show_overview(df):
    st.subheader("Top 10 Medications by Quantity")
    
    # Calculate top medications and sort by quantity
    top_meds = df.groupby(['Med_ID', 'Description'])['Quantity'].sum().reset_index()
    top_meds = top_meds.sort_values('Quantity', ascending=False).head(10)
    
    # Create a copy for the table display with formatted numbers
    top_meds_display = top_meds.copy()
    top_meds_display['Quantity'] = top_meds_display['Quantity'].map('{:,.0f}'.format)
    
    # Display table
    st.dataframe(
        top_meds_display,
        column_config={
            "Med_ID": "Medication ID",
            "Description": "Medication Name",
            "Quantity": "Total Quantity"
        },
        hide_index=True
    )
    
    # Create vertical bar chart with bars next to each other
    fig = px.bar(
        top_meds,
        x='Med_ID',
        y='Quantity',
        text='Quantity',
        title='Top 10 Medications by Receiving Quantity',
        hover_data=['Description'],
        category_orders={"Med_ID": top_meds['Med_ID'].tolist()}  # Preserve the sort order
    )
    
    # Customize the chart
    fig.update_traces(
        texttemplate='%{text:,.0f}',
        textposition='outside',
        marker_color='#1f77b4',
        width=0.6  # Make bars slightly thinner
    )
    
    fig.update_layout(
        height=500,
        xaxis_title="Medication ID",
        yaxis_title="Quantity",
        showlegend=False,
        xaxis={
            'tickangle': 45,
            'type': 'category',  # Treat x-axis as categories
            'categoryorder': 'array',  # Use custom order
            'categoryarray': top_meds['Med_ID'].tolist()  # Order by quantity
        },
        bargap=0.2  # Reduce gap between bars
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

def analyze_monthly_trends(df, min_avg_quantity=50):
    """Analyze monthly trends and identify medications needing adjustment"""
    # Convert Date to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    
    # Calculate monthly averages for each medication
    monthly_avg = df.groupby(['Med_ID', 'Description', 'Month'])['Quantity'].agg([
        'mean',
        'sum',
        'count'
    ]).reset_index()
    
    # Calculate overall average for comparison
    overall_avg = df.groupby(['Med_ID', 'Description'])['Quantity'].mean().reset_index()
    overall_avg = overall_avg.rename(columns={'Quantity': 'overall_mean'})
    
    # Filter medications with average monthly quantity > min_avg_quantity
    high_volume_meds = overall_avg[overall_avg['overall_mean'] > min_avg_quantity]
    
    # Merge monthly and overall averages
    monthly_analysis = pd.merge(monthly_avg, overall_avg, on=['Med_ID', 'Description'])
    
    # Filter for high volume medications only
    monthly_analysis = monthly_analysis[
        monthly_analysis['Med_ID'].isin(high_volume_meds['Med_ID'])
    ]
    
    # Calculate percentage difference from overall average
    monthly_analysis['pct_diff'] = ((monthly_analysis['mean'] - monthly_analysis['overall_mean']) 
                                   / monthly_analysis['overall_mean'] * 100)
    
    return monthly_analysis

def show_trends(df):
    """Display clickable month boxes and their results"""
    st.subheader("Monthly PAR Level Adjustment Alerts")
    st.info("Click on a month to see detailed PAR level adjustment recommendations")
    
    # Get monthly analysis
    monthly_analysis = analyze_monthly_trends(df)
    
    # Create month names for display
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    
    # Initialize selected month in session state if not exists
    if 'selected_month' not in st.session_state:
        st.session_state.selected_month = None
    
    # Create 3 rows of 4 columns for months
    for row in range(3):
        cols = st.columns(4)
        for col in range(4):
            month_num = row * 4 + col + 1
            month_name = month_names[month_num]
            
            with cols[col]:
                # Count variations for the month
                month_variations = len(monthly_analysis[
                    (monthly_analysis['Month'] == month_num) & 
                    (abs(monthly_analysis['pct_diff']) > 20)
                ])
                
                # Create clickable box
                if st.button(
                    f"{month_name} ({month_variations} alerts)",
                    key=f"month_{month_num}",
                    use_container_width=True,
                ):
                    st.session_state.selected_month = month_num
    
    # Display results for selected month if one is selected
    if st.session_state.selected_month:
        selected_month = st.session_state.selected_month
        st.markdown(f"### {month_names[selected_month]} Details")
        
        # Filter data for selected month
        month_data = monthly_analysis[
            (monthly_analysis['Month'] == selected_month) & 
            (abs(monthly_analysis['pct_diff']) > 20)
        ]
        
        if len(month_data) > 0:
            # Create two columns for increase/decrease recommendations
            increase_col, decrease_col = st.columns(2)
            
            # Filter and sort data for increases and decreases
            increases = month_data[month_data['pct_diff'] > 20].sort_values('pct_diff', ascending=False)
            decreases = month_data[month_data['pct_diff'] < -20].sort_values('pct_diff')
            
            # Display increases in left column (highest percentage first)
            with increase_col:
                st.markdown("#### 🔺 Increase PAR")
                if not increases.empty:
                    for _, row in increases.iterrows():
                        st.markdown(
                            f"""
                            <div style='background-color: #ff57221a; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                                <b>{row['Med_ID']}</b><br>
                                <small>{row['Description']}</small><br>
                                <b>{row['pct_diff']:.1f}% above average</b><br>
                                Average Quantity: {row['mean']:.0f}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No medications need PAR increase")
            
            # Display decreases in right column (largest decrease first)
            with decrease_col:
                st.markdown("#### 🔻 Decrease PAR")
                if not decreases.empty:
                    for _, row in decreases.iterrows():
                        st.markdown(
                            f"""
                            <div style='background-color: #1e88e51a; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                                <b>{row['Med_ID']}</b><br>
                                <small>{row['Description']}</small><br>
                                <b>{abs(row['pct_diff']):.1f}% below average</b><br>
                                Average Quantity: {row['mean']:.0f}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No medications need PAR decrease")
        else:
            st.info("No significant variations for this month")

def show_analysis1(df):
    st.subheader("Analysis 1")
    # Add your first analysis code here
    st.info("Analysis 1 will be implemented here")

def show_analysis2(df):
    st.subheader("Analysis 2")
    # Add your second analysis code here
    st.info("Analysis 2 will be implemented here")

def main():
    st.title("📦 Receiving Data Analysis")
    
    # Initialize session state for data
    if 'receiving_data' not in st.session_state:
        st.session_state.receiving_data = None
    
    # Add report type selection
    report_type = st.selectbox(
        "Select Report Type",
        options=["BD Carousel", "Omnicell"],
        index=0,
        help="Choose the type of receiving report you want to analyze"
    )
    
    if report_type == "BD Carousel":
        st.info("Upload BD Carousel receiving report to analyze the data.")
    elif report_type == "Omnicell":
        st.info("Upload Omnicell receiving report. Make sure the data follows the Omnicell format.")
    
    uploaded_file = st.file_uploader(f"Upload {report_type} Data (Excel file)", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        if report_type == "BD Carousel":
            try:
                # Load and clean BD Carousel data
                df = pd.read_excel(uploaded_file)
                df_cleaned = clean_bd_carousel_data(df)
                
                if df_cleaned is not None:
                    # Store data in session state
                    st.session_state.receiving_data = df_cleaned
                    
                    # Create tabs for different analyses
                    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trends", "Analysis 1", "Analysis 2"])
                    
                    with tab1:
                        show_overview(df_cleaned)
                    
                    with tab2:
                        show_trends(df_cleaned)
                    
                    with tab3:
                        show_analysis1(df_cleaned)
                    
                    with tab4:
                        show_analysis2(df_cleaned)
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                
        elif report_type == "Omnicell":
            st.info("Omnicell processing will be implemented based on the format")

if __name__ == "__main__":
    main() 