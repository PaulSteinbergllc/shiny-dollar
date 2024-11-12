import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Receiving Trends", page_icon="📈", layout="wide")

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

def display_monthly_alerts(monthly_data):
    """Display monthly alert boxes for PAR level adjustments"""
    st.subheader("Monthly PAR Level Adjustment Alerts")
    st.info("Showing medications with average monthly quantity > 50 and variations > 20% from mean")
    
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
                month_variations = len(monthly_data[
                    (monthly_data['Month'] == month_num) & 
                    (abs(monthly_data['pct_diff']) > 20)
                ])
                
                # Create clickable box
                if st.button(
                    f"{month_name}\n({month_variations} alerts)",
                    key=f"month_{month_num}",
                    use_container_width=True,
                ):
                    st.session_state.selected_month = month_num
    
    # Display results for selected month
    if st.session_state.selected_month:
        selected_month = st.session_state.selected_month
        st.markdown(f"### {month_names[selected_month]} Details")
        
        # Filter data for selected month
        month_data = monthly_data[
            (monthly_data['Month'] == selected_month) & 
            (abs(monthly_data['pct_diff']) > 20)
        ]
        
        if len(month_data) > 0:
            for _, row in month_data.iterrows():
                if row['pct_diff'] > 20:
                    st.markdown(
                        f"""
                        <div style='background-color: #ff57221a; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                            🔺 <b>{row['Med_ID']}</b><br>
                            <small>{row['Description']}</small><br>
                            Increase PAR<br>
                            {row['pct_diff']:.1f}% above average<br>
                            Average Quantity: {row['mean']:.0f}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                elif row['pct_diff'] < -20:
                    st.markdown(
                        f"""
                        <div style='background-color: #1e88e51a; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                            🔻 <b>{row['Med_ID']}</b><br>
                            <small>{row['Description']}</small><br>
                            Decrease PAR<br>
                            {abs(row['pct_diff']):.1f}% below average<br>
                            Average Quantity: {row['mean']:.0f}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
        else:
            st.markdown("No significant variations")

def main():
    st.title("📈 Receiving Trends Analysis")
    
    if 'receiving_data' not in st.session_state or st.session_state.receiving_data is None:
        st.warning("Please upload data from the main Receiving page first.")
        return
    
    if 'selected_month' not in st.session_state:
        st.info("Please select a month from the main Receiving page")
        return
        
    df = st.session_state.receiving_data
    selected_month = st.session_state.selected_month
    
    # Get month name
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    
    st.header(f"PAR Level Adjustments for {month_names[selected_month]}")
    
    # Analyze monthly trends
    monthly_analysis = analyze_monthly_trends(df)
    
    # Filter data for selected month
    month_data = monthly_analysis[
        (monthly_analysis['Month'] == selected_month) & 
        (abs(monthly_analysis['pct_diff']) > 20)
    ]
    
    if len(month_data) > 0:
        for _, row in month_data.iterrows():
            if row['pct_diff'] > 20:
                st.markdown(
                    f"""
                    <div style='background-color: #ff57221a; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                        🔺 <b>{row['Med_ID']}</b><br>
                        <small>{row['Description']}</small><br>
                        Increase PAR<br>
                        {row['pct_diff']:.1f}% above average<br>
                        Average Quantity: {row['mean']:.0f}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            elif row['pct_diff'] < -20:
                st.markdown(
                    f"""
                    <div style='background-color: #1e88e51a; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                        🔻 <b>{row['Med_ID']}</b><br>
                        <small>{row['Description']}</small><br>
                        Decrease PAR<br>
                        {abs(row['pct_diff']):.1f}% below average<br>
                        Average Quantity: {row['mean']:.0f}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
    else:
        st.info("No significant variations for this month")

if __name__ == "__main__":
    main() 