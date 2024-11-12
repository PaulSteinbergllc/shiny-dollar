import streamlit as st
import plotly.express as px
import pandas as pd
import requests
import logging
from typing import Dict
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

logger = logging.getLogger(__name__)

# Configure the page
st.set_page_config(
    page_title="Analysis",
    page_icon="📊",
    layout="wide"
)

class NDCApi:
    def __init__(self):
        self.base_url = "https://api.fda.gov/drug/drugsfda.json"

    def convert_11_to_10_digit_ndcs(self, ndc: str) -> list:
        """Convert 11-digit NDC (5-4-2) to possible 10-digit formats (4-4-2, 5-3-2, 5-4-1)"""
        ndc_clean = ndc.replace('-', '').replace(' ', '').zfill(11)
        
        formats = []
        
        if ndc_clean[0] == '0':
            formats.append(f"{ndc_clean[1:5]}-{ndc_clean[5:9]}-{ndc_clean[9:11]}")
        
        if ndc_clean[5] == '0':
            formats.append(f"{ndc_clean[:5]}-{ndc_clean[6:9]}-{ndc_clean[9:11]}")
        
        if ndc_clean[9] == '0':
            formats.append(f"{ndc_clean[:5]}-{ndc_clean[5:9]}-{ndc_clean[10]}")
        
        return formats

    def get_medication_info(self, ndc: str) -> dict:
        """Get medication name and strength from FDA API"""
        try:
            possible_formats = self.convert_11_to_10_digit_ndcs(ndc)
            
            for ndc_format in possible_formats:
                url = f"{self.base_url}?search=openfda.package_ndc:\"{ndc_format}\"&limit=1"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('results'):
                        result = data['results'][0]
                        if result.get('products'):
                            product = result['products'][0]
                            if product.get('active_ingredients'):
                                active_ingredient = product['active_ingredients'][0]
                                return {
                                    "name": product.get('brand_name', 'Name not found'),
                                    "strength": active_ingredient.get('strength', 'Strength not found')
                                }
            
            return {
                "name": "Name not found",
                "strength": "Strength not found"
            }
                
        except Exception as e:
            logger.error(f"Error fetching medication info: {e}")
            return {
                "name": "Error",
                "strength": "Error"
            }

def format_ndc(ndc):
    """Format NDC into 5-4-2 format"""
    # First remove any existing dashes
    ndc = str(ndc).replace('-', '')
    # Pad with zeros to ensure 11 digits
    ndc = ndc.zfill(11)
    # Format as 5-4-2
    return f"{ndc[:5]}-{ndc[5:9]}-{ndc[9:]}"

def create_top_ndc_charts(df, n=10):
    """Create top NDC visualizations"""
    ndc_api = NDCApi()
    
    # Create two columns for the charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Top NDCs by Total Expenditure
        expenditure_by_ndc = (df.groupby('NDC')['Price']
                            .sum()
                            .sort_values(ascending=False)
                            .head(n)
                            .reset_index())
        
        # Format NDCs and get drug info
        expenditure_by_ndc['NDC'] = expenditure_by_ndc['NDC'].apply(format_ndc)
        med_info = expenditure_by_ndc['NDC'].apply(lambda x: ndc_api.get_medication_info(x))
        expenditure_by_ndc['Drug_Name'] = med_info.apply(lambda x: x['name'])
        expenditure_by_ndc['Strength'] = med_info.apply(lambda x: x['strength'])
        
        fig_expenditure = px.bar(
            expenditure_by_ndc,
            x='Price',
            y='NDC',
            orientation='h',
            title=f'Top {n} NDCs by Expenditure',
            labels={'Price': 'Total Expenditure ($)', 'NDC': 'NDC'},
        )
        
        # Customize hover template
        fig_expenditure.update_traces(
            hovertemplate="<br>".join([
                "Total Expenditure ($)=%{x:,.2f}",
                "NDC=%{y}",
                "Name: %{customdata[0]}",
                "Strength: %{customdata[1]}"
            ])
        )
        
        # Add drug info to hover data
        fig_expenditure.update_traces(
            customdata=expenditure_by_ndc[['Drug_Name', 'Strength']]
        )
        
        fig_expenditure.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis={'categoryorder': 'total ascending'},
            xaxis_tickformat='$,.0f',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_expenditure, use_container_width=True)
    
    with col2:
        # Top NDCs by Quantity
        quantity_by_ndc = (df.groupby('NDC')['Quantity']
                         .sum()
                         .sort_values(ascending=False)
                         .head(n)
                         .reset_index())
        
        # Format NDCs and get drug info
        quantity_by_ndc['NDC'] = quantity_by_ndc['NDC'].apply(format_ndc)
        med_info = quantity_by_ndc['NDC'].apply(lambda x: ndc_api.get_medication_info(x))
        quantity_by_ndc['Drug_Name'] = med_info.apply(lambda x: x['name'])
        quantity_by_ndc['Strength'] = med_info.apply(lambda x: x['strength'])
        
        fig_quantity = px.bar(
            quantity_by_ndc,
            x='Quantity',
            y='NDC',
            orientation='h',
            title=f'Top {n} NDCs by Quantity',
            labels={'Quantity': 'Total Quantity', 'NDC': 'NDC'}
        )
        
        # Customize hover template
        fig_quantity.update_traces(
            hovertemplate="<br>".join([
                "Total Quantity=%{x:,.0f}",
                "NDC=%{y}",
                "Name: %{customdata[0]}",
                "Strength: %{customdata[1]}"
            ])
        )
        
        # Add drug info to hover data
        fig_quantity.update_traces(
            customdata=quantity_by_ndc[['Drug_Name', 'Strength']]
        )
        
        fig_quantity.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis={'categoryorder': 'total ascending'},
            xaxis_tickformat=',d',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_quantity, use_container_width=True)

def create_trend_analysis(df):
    st.header("Trend Analysis")
    
    # Group by date and calculate daily totals
    daily_totals = df.groupby('Date').agg({
        'Price': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    # Create tabs for different trend views
    trend_tab1, trend_tab2 = st.tabs(["Expenditure Trends", "Quantity Trends"])
    
    with trend_tab1:
        fig_exp = px.line(
            daily_totals, 
            x='Date', 
            y='Price',
            title='Daily Expenditure Trends',
            labels={'Price': 'Total Expenditure ($)', 'Date': 'Date'}
        )
        fig_exp.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_tickformat='%Y-%m-%d',
            yaxis_tickformat='$,.0f'
        )
        st.plotly_chart(fig_exp, use_container_width=True)
        
    with trend_tab2:
        fig_qty = px.line(
            daily_totals, 
            x='Date', 
            y='Quantity',
            title='Daily Quantity Trends',
            labels={'Quantity': 'Total Quantity', 'Date': 'Date'}
        )
        fig_qty.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_tickformat='%Y-%m-%d',
            yaxis_tickformat=',d'
        )
        st.plotly_chart(fig_qty, use_container_width=True)

def create_price_anomalies(df_anomaly, threshold):
    """Create price anomalies analysis section"""
    st.markdown("""
    ### Understanding Price Anomalies
    
    Price anomalies are transactions where the unit price (price per quantity) significantly deviates from the 
    typical unit price for that specific medication. These could indicate:
    - Unusual price spikes or drops
    - Potential data entry errors
    - Special purchasing circumstances
    - Market fluctuations
    """)
    
    # Calculate average unit price per NDC
    avg_unit_prices = df_anomaly.groupby('NDC')['Unit_Price'].mean()
    std_unit_prices = df_anomaly.groupby('NDC')['Unit_Price'].std()
    
    # Calculate Z-scores per NDC (comparing against each NDC's own average unit price)
    df_anomaly['Unit_Price_ZScore'] = df_anomaly.apply(
        lambda row: (row['Unit_Price'] - avg_unit_prices[row['NDC']]) / std_unit_prices[row['NDC']]
        if row['NDC'] in std_unit_prices and std_unit_prices[row['NDC']] != 0
        else 0,
        axis=1
    )
    
    # Identify price anomalies
    price_anomalies = df_anomaly[abs(df_anomaly['Unit_Price_ZScore']) > threshold].copy()
    
    if not price_anomalies.empty:
        # Format and prepare anomalies data
        price_anomalies['Date'] = price_anomalies['Date'].dt.strftime('%Y-%m-%d')
        price_anomalies['Unit_Price'] = price_anomalies['Unit_Price'].round(2)
        price_anomalies['Z-Score'] = price_anomalies['Unit_Price_ZScore'].round(2)
        price_anomalies['Avg_Unit_Price'] = price_anomalies['NDC'].map(avg_unit_prices).round(2)
        price_anomalies['Price_Difference'] = ((price_anomalies['Unit_Price'] - price_anomalies['Avg_Unit_Price']) / price_anomalies['Avg_Unit_Price'] * 100).round(1)
        
        # Get medication info for each NDC
        ndc_api = NDCApi()
        med_info = price_anomalies['NDC'].apply(lambda x: ndc_api.get_medication_info(x))
        price_anomalies['Medication'] = med_info.apply(lambda x: x['name'])
        
        # Group anomalies by NDC and Medication
        grouped_anomalies = price_anomalies.groupby(['NDC', 'Medication']).agg({
            'Unit_Price': 'count',
            'Z-Score': 'mean'
        }).reset_index()
        grouped_anomalies = grouped_anomalies.rename(columns={'Unit_Price': 'Number of Anomalies'})
        
        # Display summary of anomalies by medication
        st.markdown("### Select a Medication to View Detailed Anomalies")
        
        # Create clickable medication list
        for _, row in grouped_anomalies.iterrows():
            if st.button(
                f"{row['Medication']} (NDC: {row['NDC']}) - {row['Number of Anomalies']} anomalies",
                key=f"med_{row['NDC']}"
            ):
                st.markdown(f"#### Detailed Anomalies for {row['Medication']}")
                
                # Filter and display anomalies for selected medication
                med_anomalies = price_anomalies[price_anomalies['NDC'] == row['NDC']]
                st.dataframe(
                    med_anomalies[[
                        'Date', 'NDC', 'Medication', 'Unit_Price', 
                        'Avg_Unit_Price', 'Price_Difference', 'Quantity', 'Z-Score'
                    ]].sort_values('Z-Score', ascending=False),
                    column_config={
                        'Date': 'Date',
                        'NDC': 'NDC',
                        'Medication': 'Medication Name',
                        'Unit_Price': st.column_config.NumberColumn('Unit Price ($)', format='$%.2f'),
                        'Avg_Unit_Price': st.column_config.NumberColumn('Avg Unit Price ($)', format='$%.2f'),
                        'Price_Difference': st.column_config.NumberColumn('% Difference from Average', format='%.1f%%'),
                        'Quantity': 'Quantity',
                        'Z-Score': 'Deviation Score'
                    },
                    hide_index=True
                )
                
                # Create visualization for selected medication
                fig_price = px.scatter(
                    df_anomaly[df_anomaly['NDC'] == row['NDC']],
                    x='Date',
                    y='Unit_Price',
                    title=f'Unit Price Anomalies for {row["Medication"]}',
                    labels={'Unit_Price': 'Unit Price ($)', 'Date': 'Date'}
                )
                
                # Add horizontal line for average unit price
                fig_price.add_hline(
                    y=avg_unit_prices[row['NDC']],
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Average Unit Price"
                )
                
                # Add anomaly points
                fig_price.add_scatter(
                    x=med_anomalies['Date'],
                    y=med_anomalies['Unit_Price'],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=10)
                )
                
                fig_price.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    title={
                        'text': f'Unit Price Anomalies for {row["Medication"]}<br><sup>Red dots indicate anomalous transactions</sup>',
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    }
                )
                st.plotly_chart(fig_price, use_container_width=True)
                
                st.markdown(f"""
                ### How to Interpret This Chart
                
                - **Gray dots**: Regular transactions with normal unit prices
                - **Red dots**: Anomalous transactions where unit price deviates significantly
                - **Green dashed line**: Average unit price (${avg_unit_prices[row['NDC']]:.2f})
                - **X-axis**: Timeline of transactions
                - **Y-axis**: Unit price (price per quantity) in dollars
                """)
    else:
        st.info("No price anomalies detected with the current threshold. Try lowering the threshold to detect more anomalies.")

def create_quantity_anomalies(df_anomaly, threshold):
    """Create quantity anomalies analysis section"""
    st.markdown("""
    ### Understanding Quantity Anomalies
    
    Quantity anomalies are transactions where the ordered quantity significantly deviates from the 
    typical order quantity for that specific medication. These could indicate:
    - Unusual ordering patterns
    - Potential data entry errors
    - Special ordering circumstances
    - Changes in demand
    """)
    
    # Calculate average quantity per NDC
    avg_quantities = df_anomaly.groupby('NDC')['Quantity'].mean()
    std_quantities = df_anomaly.groupby('NDC')['Quantity'].std()
    
    # Calculate Z-scores per NDC
    df_anomaly['Quantity_ZScore'] = df_anomaly.apply(
        lambda row: (row['Quantity'] - avg_quantities[row['NDC']]) / std_quantities[row['NDC']]
        if row['NDC'] in std_quantities and std_quantities[row['NDC']] != 0
        else 0,
        axis=1
    )
    
    # Identify quantity anomalies
    quantity_anomalies = df_anomaly[abs(df_anomaly['Quantity_ZScore']) > threshold].copy()
    
    if not quantity_anomalies.empty:
        # Format and prepare anomalies data
        quantity_anomalies['Date'] = quantity_anomalies['Date'].dt.strftime('%Y-%m-%d')
        quantity_anomalies['Z-Score'] = quantity_anomalies['Quantity_ZScore'].round(2)
        quantity_anomalies['Avg_Quantity'] = quantity_anomalies['NDC'].map(avg_quantities).round(0)
        quantity_anomalies['Quantity_Difference'] = (
            (quantity_anomalies['Quantity'] - quantity_anomalies['Avg_Quantity']) / 
            quantity_anomalies['Avg_Quantity'] * 100
        ).round(1)
        
        # Get medication info
        ndc_api = NDCApi()
        med_info = quantity_anomalies['NDC'].apply(lambda x: ndc_api.get_medication_info(x))
        quantity_anomalies['Medication'] = med_info.apply(lambda x: x.get('name', 'Unknown'))
        
        # Group anomalies
        grouped_anomalies = quantity_anomalies.groupby(['NDC', 'Medication']).agg({
            'Quantity': 'count',
            'Z-Score': 'mean'
        }).reset_index()
        grouped_anomalies = grouped_anomalies.rename(columns={'Quantity': 'Number of Anomalies'})
        
        st.markdown("### Select a Medication to View Detailed Anomalies")
        
        for _, row in grouped_anomalies.iterrows():
            if st.button(
                f"{row['Medication']} (NDC: {row['NDC']}) - {row['Number of Anomalies']} anomalies",
                key=f"qty_med_{row['NDC']}"
            ):
                st.markdown(f"#### Detailed Anomalies for {row['Medication']}")
                
                med_anomalies = quantity_anomalies[quantity_anomalies['NDC'] == row['NDC']]
                st.dataframe(
                    med_anomalies[[
                        'Date', 'Quantity', 'Avg_Quantity', 
                        'Quantity_Difference', 'Z-Score'
                    ]].style.format({
                        'Quantity': '{:,.0f}',
                        'Avg_Quantity': '{:,.0f}',
                        'Quantity_Difference': '{:+.1f}%',
                        'Z-Score': '{:+.2f}'
                    }),
                    hide_index=True
                )
                
                # Create visualization
                fig_quantity = px.scatter(
                    df_anomaly[df_anomaly['NDC'] == row['NDC']],
                    x='Date',
                    y='Quantity',
                    title=f'Quantity Anomalies for {row["Medication"]}',
                    labels={'Quantity': 'Order Quantity', 'Date': 'Date'}
                )
                
                # Add average quantity line
                fig_quantity.add_hline(
                    y=avg_quantities[row['NDC']],
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Average Order Quantity"
                )
                
                # Add anomaly points
                fig_quantity.add_scatter(
                    x=med_anomalies['Date'],
                    y=med_anomalies['Quantity'],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=10)
                )
                
                fig_quantity.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    title={
                        'text': f'Quantity Anomalies for {row["Medication"]}<br><sup>Red dots indicate anomalous orders</sup>',
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    }
                )
                st.plotly_chart(fig_quantity, use_container_width=True)
                
                st.markdown(f"""
                ### How to Interpret This Chart
                
                - **Gray dots**: Regular orders with normal quantities
                - **Red dots**: Anomalous orders where quantity deviates significantly
                - **Green dashed line**: Average order quantity ({int(avg_quantities[row['NDC']])})
                - **X-axis**: Timeline of orders
                - **Y-axis**: Order quantity
                
                ### What Makes a Quantity Anomalous?
                
                An order is flagged as anomalous when its quantity deviates significantly from the typical 
                order quantity for this medication. The current threshold is set to {threshold} standard deviations 
                from the mean, which means these orders are unusually large or small compared to normal ordering patterns.
                """)
    else:
        st.info("No quantity anomalies detected with the current threshold. Try lowering the threshold to detect more anomalies.")

def create_anomaly_analysis(df):
    st.header("Anomaly Detection")
    
    # Calculate unit price for each transaction
    df_anomaly = df.copy()
    df_anomaly['Unit_Price'] = df_anomaly['Price'] / df_anomaly['Quantity']
    
    threshold = st.slider(
        "Select Z-Score threshold for anomalies", 
        2.0, 5.0, 3.0, 0.5,
        help="Z-score measures how many standard deviations a value is from the mean for that specific medication."
    )
    
    # Create tabs for different anomaly views
    anomaly_tab1, anomaly_tab2 = st.tabs(["Price Anomalies", "Quantity Anomalies"])
    
    with anomaly_tab1:
        create_price_anomalies(df_anomaly, threshold)
    
    with anomaly_tab2:
        create_quantity_anomalies(df_anomaly, threshold)

def create_top_ndc_tab(df):
    st.subheader("📈 Top NDC Analysis")
    
    # Add filter for number of NDCs
    top_n = st.slider("Select number of top NDCs", 5, 20, 10)
    
    # Initialize NDC API
    ndc_api = NDCApi()
    
    # Create two columns for the charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Top NDCs by Total Expenditure
        expenditure_by_ndc = (df.groupby('NDC')['Price']
                            .sum()
                            .sort_values(ascending=False)
                            .head(top_n)
                            .reset_index())
        
        # Format NDCs and get medication info
        expenditure_by_ndc['NDC_Formatted'] = expenditure_by_ndc['NDC'].apply(format_ndc)
        med_info = expenditure_by_ndc['NDC'].apply(lambda x: ndc_api.get_medication_info(x))
        expenditure_by_ndc['Medication'] = med_info.apply(lambda x: x['name'])
        expenditure_by_ndc['Strength'] = med_info.apply(lambda x: x['strength'])
        
        # Create hover text
        expenditure_by_ndc['hover_text'] = (
            'NDC: ' + expenditure_by_ndc['NDC_Formatted'] + '<br>' +
            'Medication: ' + expenditure_by_ndc['Medication'] + '<br>' +
            'Strength: ' + expenditure_by_ndc['Strength'] + '<br>' +
            'Total Expenditure: $' + expenditure_by_ndc['Price'].round(2).astype(str)
        )
        
        fig_expenditure = px.bar(
            expenditure_by_ndc,
            x='NDC_Formatted',
            y='Price',
            title=f'Top {top_n} NDCs by Expenditure',
            labels={'Price': 'Total Expenditure ($)', 'NDC_Formatted': 'NDC Code'},
            custom_data=['hover_text']
        )
        
        fig_expenditure.update_traces(
            hovertemplate="%{customdata[0]}<extra></extra>"
        )
        
        st.plotly_chart(fig_expenditure, use_container_width=True)
    
    with col2:
        # Top NDCs by Quantity
        quantity_by_ndc = (df.groupby('NDC')['Quantity']
                         .sum()
                         .sort_values(ascending=False)
                         .head(top_n)
                         .reset_index())
        
        # Format NDCs and get medication info
        quantity_by_ndc['NDC_Formatted'] = quantity_by_ndc['NDC'].apply(format_ndc)
        med_info = quantity_by_ndc['NDC'].apply(lambda x: ndc_api.get_medication_info(x))
        quantity_by_ndc['Medication'] = med_info.apply(lambda x: x['name'])
        quantity_by_ndc['Strength'] = med_info.apply(lambda x: x['strength'])
        
        # Create hover text
        quantity_by_ndc['hover_text'] = (
            'NDC: ' + quantity_by_ndc['NDC_Formatted'] + '<br>' +
            'Medication: ' + quantity_by_ndc['Medication'] + '<br>' +
            'Strength: ' + quantity_by_ndc['Strength'] + '<br>' +
            'Total Quantity: ' + quantity_by_ndc['Quantity'].astype(str)
        )
        
        fig_quantity = px.bar(
            quantity_by_ndc,
            x='NDC_Formatted',
            y='Quantity',
            title=f'Top {top_n} NDCs by Quantity',
            labels={'Quantity': 'Total Quantity', 'NDC_Formatted': 'NDC Code'},
            custom_data=['hover_text']
        )
        
        fig_quantity.update_traces(
            hovertemplate="%{customdata[0]}<extra></extra>"
        )
        
        st.plotly_chart(fig_quantity, use_container_width=True)

def create_trend_tab(df):
    st.subheader("📊 Trend Analysis")
    
    # Prepare monthly data
    monthly_data = df.groupby(df['Date'].dt.to_period('M')).agg({
        'Price': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    monthly_data['Date'] = monthly_data['Date'].dt.to_timestamp()
    
    # Check data length
    n_months = len(monthly_data)
    if n_months < 24:
        st.warning(f"""
            ⚠️ Trend analysis requires at least 24 months of data for seasonal decomposition.
            Your dataset contains {n_months} months of data.
            
            Showing basic trend visualization instead.
        """)
        
        # Create simple trend visualization
        fig_price = px.line(
            monthly_data,
            x='Date',
            y='Price',
            title='Monthly Expenditure Trend'
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        fig_quantity = px.line(
            monthly_data,
            x='Date',
            y='Quantity',
            title='Monthly Quantity Trend'
        )
        st.plotly_chart(fig_quantity, use_container_width=True)
        
        return
        
    # If we have enough data, perform seasonal decomposition
    try:
        decomposition = seasonal_decompose(
            monthly_data['Price'], 
            period=12, 
            extrapolate_trend='freq'
        )
        
        # Plot components
        fig1 = px.line(x=monthly_data['Date'], y=decomposition.trend, title="Trend Component")
        fig2 = px.line(x=monthly_data['Date'], y=decomposition.seasonal, title="Seasonal Component")
        fig3 = px.line(x=monthly_data['Date'], y=decomposition.resid, title="Residual Component")
        
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error performing trend analysis: {str(e)}")

def create_anomaly_tab(df):
    st.subheader("🔍 Anomaly Analysis")
    create_anomaly_analysis(df)

def main():
    st.title("📊 Analysis")
    
    # Check if analysis has been run
    if 'analysis_run' not in st.session_state or not st.session_state.analysis_run:
        st.warning("Please run the analysis from the Home page first.")
        return
    
    # Get processed data
    df = st.session_state.processed_df
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Top NDC", "Trend", "Anomaly"])
    
    with tab1:
        create_top_ndc_tab(df)
    
    with tab2:
        create_trend_tab(df)
        
    with tab3:
        create_anomaly_tab(df)

if __name__ == "__main__":
    main()
