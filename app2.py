import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, Any
import plotly.graph_objects as go

class DataUploadInterface:
    def __init__(self):
        st.set_page_config(
            page_title="Pharmacy Purchase Data Analysis",
            page_icon="💊",
            layout="wide"
        )
        self.years = list(range(2024, 1989, -1))
        self.required_columns = ['Date', 'Product_ID', 'Quantity', 'Price']
        
        # Define year groups as class variable
        self.year_groups = {
            "2021-2024": list(range(2024, 2020, -1)),
            "2011-2020": list(range(2020, 2010, -1)),
            "2001-2010": list(range(2010, 2000, -1)),
            "1990-2000": list(range(2000, 1989, -1))
        }
        
        # Initialize session states
        if 'active_group' not in st.session_state:
            st.session_state.active_group = None
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = {}
        if 'current_mapping' not in st.session_state:
            st.session_state.current_mapping = None
        if 'show_mapper' not in st.session_state:
            st.session_state.show_mapper = False
        if 'temp_df' not in st.session_state:
            st.session_state.temp_df = None
        if 'temp_year' not in st.session_state:
            st.session_state.temp_year = None
        if 'previous_mapping' not in st.session_state:
            st.session_state.previous_mapping = None
        if 'show_analysis' not in st.session_state:
            st.session_state.show_analysis = False
        if 'show_abc' not in st.session_state:
            st.session_state.show_abc = False
        if 'show_trends' not in st.session_state:
            st.session_state.show_trends = False
        if 'show_anomalies' not in st.session_state:
            st.session_state.show_anomalies = False

    def create_interface(self):
        """Create the main interface"""
        # Add custom CSS
        st.markdown("""
            <style>
            .streamlit-expanderHeader {
                background-color: #f0f2f6;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-bottom: 10px;
            }
            .streamlit-expanderHeader:hover {
                background-color: #e8eaf6;
            }
            .stExpander {
                border: none !important;
            }
            .upload-row {
                padding: 10px 0;
                border-bottom: 1px solid #eee;
            }
            .year-label {
                font-weight: bold;
                color: #2c3e50;
            }
            .stButton button {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 0.5rem 2rem;
            }
            .stButton button:hover {
                background-color: #45a049;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.title("Pharmacy Purchase Data Upload")
        st.markdown("---")

        # Initialize show_analysis state if not exists
        if 'show_analysis' not in st.session_state:
            st.session_state.show_analysis = False

        if st.session_state.show_analysis:
            self._show_analysis_page()
        elif st.session_state.show_mapper:
            self._show_mapping_interface()
        elif st.session_state.show_trends:
            self._show_trend_analysis()
        elif st.session_state.show_anomalies:
            self._show_anomaly_detection()
        elif st.session_state.show_abc:
            self._show_abc_analysis()
        else:
            self._create_upload_section()

    def _create_upload_section(self):
        """Create the file upload section"""
        st.subheader("Upload Purchase Data")
        st.markdown("""
        Please upload your purchase data Excel files for each year.
        
        **Required format:**
        - Excel file (.xlsx or .xls)
        - Required columns: Date, NDC, Quantity, Price
        """)

        # Create expanders for each year group
        for group_name, years in self.year_groups.items():
            is_expanded = (st.session_state.active_group == group_name or 
                          (st.session_state.active_group is None and 
                           any(year in st.session_state.uploaded_files for year in years)))
            
            with st.expander(group_name, expanded=is_expanded):
                if is_expanded and st.session_state.active_group is None:
                    st.session_state.active_group = group_name
                
                for year in years:
                    self._create_year_upload_row(year)
        
        # Add Run button after all expanders
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Run Analysis", type="primary", use_container_width=True):
                if st.session_state.uploaded_files:
                    st.session_state.show_analysis = True
                    st.rerun()
                else:
                    st.error("Please upload at least one file before running analysis")

    def _create_summary_section(self):
        """Create the data summary section"""
        st.markdown("## Data Summary")
        
        if st.session_state.uploaded_files:
            # Calculate total records
            total_records = sum(len(data['mapped_df']) for data in st.session_state.uploaded_files.values())
            st.markdown(f"**Total Records:** {total_records:,}")
            
            try:
                # Combine all mapped dataframes
                all_data = pd.concat([
                    data['mapped_df'] for data in st.session_state.uploaded_files.values()
                ])
                
                # Ensure proper data types
                all_data['Date'] = pd.to_datetime(all_data['Date'])
                all_data['Quantity'] = pd.to_numeric(all_data['Quantity'], errors='coerce')
                all_data['Price'] = pd.to_numeric(all_data['Price'], errors='coerce')
                
                # Calculate summary statistics
                summary_stats = {
                    "Date Range": f"{all_data['Date'].min().strftime('%Y-%m-%d')} to {all_data['Date'].max().strftime('%Y-%m-%d')}",
                    "Total NDCs": f"{all_data['NDC'].nunique():,}",
                    "Total Quantity": f"{all_data['Quantity'].sum():,.0f}",
                    "Total Value": f"${all_data['Price'].sum():,.2f}",
                    "Average Price": f"${all_data['Price'].mean():,.2f}"
                }
                
                # Display summary statistics
                for label, value in summary_stats.items():
                    st.markdown(f"**{label}:** {value}")
                    
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.markdown("Please ensure all data is properly formatted:")
                st.markdown("- Date should be in a valid date format")
                st.markdown("- Quantity should be numeric")
                st.markdown("- Price should be numeric")

    def _create_year_upload_row(self, year: int):
        """Create a row for each year's file upload"""
        if f'show_preview_{year}' not in st.session_state:
            st.session_state[f'show_preview_{year}'] = False

        col1, col2, col3, col4, col5 = st.columns([1, 2, 0.5, 0.5, 0.5])
        
        with col1:
            st.markdown(f"**{year}**")
        
        with col2:
            if year in st.session_state.uploaded_files:
                st.markdown(
                    f"""
                    <div style="
                        padding: 8px; 
                        background-color: #e8f0fe; 
                        border-radius: 5px; 
                        border: 1px solid #ccc;
                        text-align: center;
                    ">
                        ✅ File uploaded ({len(st.session_state.uploaded_files[year]['mapped_df'])} records)
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                file = st.file_uploader(
                    f"Upload {year} data",
                    type=['xlsx', 'xls'],
                    key=f"file_{year}",
                    label_visibility="collapsed"
                )
                
                if file is not None:
                    try:
                        df = pd.read_excel(file)
                        st.session_state.temp_df = df
                        st.session_state.temp_year = year
                        st.session_state.show_mapper = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Action buttons for uploaded files
        if year in st.session_state.uploaded_files:
            with col3:
                # Toggle preview without page refresh
                if st.button("View", key=f"view_{year}", help="View mapped data"):
                    st.session_state[f'show_preview_{year}'] = not st.session_state[f'show_preview_{year}']
                    # Store the current group as active
                    st.session_state.active_group = next(
                        group_name for group_name, years in self.year_groups.items() 
                        if year in years
                    )
                    st.rerun()
            
            with col4:
                if st.button("✏️", key=f"edit_{year}", help="Edit mapping"):
                    st.session_state.temp_df = st.session_state.uploaded_files[year]['original_df']
                    st.session_state.temp_year = year
                    st.session_state.show_mapper = True
                    st.session_state.previous_mapping = st.session_state.uploaded_files[year]['mapping']
                    st.rerun()
            
            with col5:
                if st.button("❌", key=f"delete_{year}", help="Remove this file"):
                    if st.session_state.uploaded_files.pop(year, None):
                        st.session_state[f'show_preview_{year}'] = False
                        st.success(f"File for year {year} removed")
                        st.rerun()
            
            # Show preview if enabled
            if st.session_state[f'show_preview_{year}']:
                st.markdown("### Preview of Mapped Data")
                st.dataframe(
                    st.session_state.uploaded_files[year]['mapped_df'].head(15),
                    use_container_width=True,
                    height=400
                )

    def _format_ndc(self, ndc_number):
        """Maintain and format NDC in 5-4-2 format"""
        try:
            # Convert to string and remove any hyphens first
            ndc = str(ndc_number).replace('-', '')
            
            # Format as 5-4-2
            formatted_ndc = f"{ndc[:5]}-{ndc[5:9]}-{ndc[9:11]}"
            return formatted_ndc
        except Exception:
            return ndc_number

    def _validate_ndc(self, ndc):
        """Validate NDC format (5-4-2)"""
        try:
            parts = str(ndc).split('-')
            valid = (len(parts) == 3 and 
                    len(parts[0]) == 5 and 
                    len(parts[1]) == 4 and 
                    len(parts[2]) == 2)
            return valid
        except:
            return False

    def _check_column_match(self, df, previous_mapping):
        """Check which columns from previous mapping exist in new file"""
        file_columns = set(df.columns)
        status = {}
        auto_mapping = {}
        
        for req_col, prev_col in previous_mapping.items():
            if prev_col in file_columns:
                status[req_col] = True  # Column exists
                auto_mapping[req_col] = prev_col
            else:
                status[req_col] = False  # Column doesn't exist
                auto_mapping[req_col] = None
                
        return status, auto_mapping

    def _show_mapping_interface(self):
        """Show the column mapping interface"""
        st.markdown("## Column Mapping")
        
        if st.button("← Back to Upload", type="secondary"):
            st.session_state.show_mapper = False
            st.session_state.temp_df = None
            st.session_state.temp_year = None
            st.rerun()
            return

        if st.session_state.temp_df is not None and st.session_state.temp_year is not None:
            df = st.session_state.temp_df
            year = st.session_state.temp_year
            file_columns = list(df.columns)
            mapping = {}
            
            # Get last successful mapping from any year
            previous_mapping = None
            if st.session_state.uploaded_files:
                # Get the most recent mapping
                last_upload = max(st.session_state.uploaded_files.items(), 
                                key=lambda x: x[1]['upload_date'])
                previous_mapping = last_upload[1]['mapping']
            
            # Check column matches if previous mapping exists
            mapping_status = {}
            auto_mapping = {}
            if previous_mapping:
                mapping_status, auto_mapping = self._check_column_match(df, previous_mapping)
            
            # Display file preview
            st.markdown("### Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Column mapping
            st.markdown("### Map Columns")
            
            # Create two columns for mapping
            col1, col2 = st.columns([1, 1])
            
            # Define mapping labels
            mapping_labels = {
                'Date': 'Date',
                'Product_ID': 'NDC',
                'Quantity': 'Quantity',
                'Price': 'Price'
            }
            
            with col1:
                st.markdown("**Select corresponding columns:**")
                for req_col in self.required_columns:
                    col_a, col_b = st.columns([4, 1])
                    
                    with col_a:
                        # Determine default selection
                        if auto_mapping and auto_mapping[req_col]:
                            default_idx = file_columns.index(auto_mapping[req_col])
                        else:
                            default_idx = file_columns.index(req_col) if req_col in file_columns else 0
                        
                        mapping[req_col] = st.selectbox(
                            f"{mapping_labels[req_col]}:",
                            options=file_columns,
                            index=default_idx,
                            key=f"map_{year}_{req_col}"
                        )
                    
                    with col_b:
                        if previous_mapping:
                            if mapping_status[req_col]:
                                st.markdown("✅")  # Green check mark
                            else:
                                st.markdown("❌")  # Red X mark
            
            with col2:
                st.markdown("**Preview Mapped Data**")
                if len(set(mapping.values())) == len(mapping):
                    try:
                        mapped_df = df.rename(columns={v: k for k, v in mapping.items()})
                        mapped_df['Product_ID'] = mapped_df['Product_ID'].apply(self._format_ndc)
                        preview_df = mapped_df[self.required_columns].head()
                        preview_df = preview_df.rename(columns={'Product_ID': 'NDC'})
                        st.dataframe(preview_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error in mapping preview: {str(e)}")
                else:
                    st.error("Each column must be mapped to a unique field")

            # Add note about automatic mapping
            if previous_mapping:
                st.info("ℹ️ Previous mapping has been applied where possible. Green check marks (✅) indicate automatically mapped columns.")

            # Confirm mapping button
            if st.button("Confirm Mapping", type="primary"):
                if len(set(mapping.values())) != len(mapping):
                    st.error("Please ensure each column is mapped to a unique field")
                else:
                    try:
                        mapped_df = df.rename(columns={v: k for k, v in mapping.items()})
                        mapped_df['Product_ID'] = mapped_df['Product_ID'].apply(self._format_ndc)
                        
                        st.session_state.uploaded_files[year] = {
                            'original_df': df,
                            'mapped_df': mapped_df[self.required_columns].rename(columns={'Product_ID': 'NDC'}),
                            'mapping': mapping,
                            'upload_date': pd.Timestamp.now()
                        }
                        
                        st.session_state.show_mapper = False
                        st.session_state.temp_df = None
                        st.session_state.temp_year = None
                        st.success("✅ Column mapping completed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error saving mapped data: {str(e)}")

    def _process_data(self):
        """Process the uploaded data"""
        try:
            # Combine all mapped dataframes
            combined_df = pd.concat([
                data['mapped_df'] for data in st.session_state.uploaded_files.values()
            ])
            
            # Display summary statistics
            st.markdown("### Data Summary")
            st.write("Total Records:", len(combined_df))
            st.write("Date Range:", combined_df['Date'].min(), "to", combined_df['Date'].max())
            st.write("Total Products:", combined_df['Product_ID'].nunique())
            
            # Create a simple visualization
            fig = go.Figure()
            monthly_sales = combined_df.groupby(pd.Grouper(key='Date', freq='M'))['Quantity'].sum()
            
            fig.add_trace(go.Scatter(
                x=monthly_sales.index,
                y=monthly_sales.values,
                mode='lines+markers',
                name='Monthly Sales'
            ))
            
            fig.update_layout(
                title="Monthly Sales Overview",
                xaxis_title="Date",
                yaxis_title="Total Quantity",
                showlegend=True
            )
            
            st.plotly_chart(fig)
            
            # Option to download processed data
            st.download_button(
                label="Download Processed Data",
                data=combined_df.to_csv(index=False).encode('utf-8'),
                file_name="processed_data.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

    def _calculate_trends(self, data):
        """Calculate trends from the data"""
        trends = {}
        
        # Monthly trends
        monthly_data = data.groupby(data['Date'].dt.to_period('M')).agg({
            'Quantity': 'sum',
            'Price': 'sum',
            'NDC': 'nunique'
        }).reset_index()
        
        # Calculate month-over-month changes
        for col in ['Quantity', 'Price', 'NDC']:
            pct_change = monthly_data[col].pct_change().iloc[-1]
            trends[f'Monthly {col} Change'] = f"{'↑' if pct_change > 0 else '↓'} {abs(pct_change):.1%}"
        
        # Top NDCs by quantity
        top_ndcs = data.groupby('NDC').agg({
            'Quantity': 'sum',
            'Price': 'sum'
        }).sort_values('Quantity', ascending=False).head(5)
        
        trends['Top NDCs'] = top_ndcs
        
        return trends

    def _detect_anomalies(self, data):
        """Detect anomalies in the data using statistical methods"""
        anomalies = {}
        
        try:
            # Ensure date is in datetime format
            data = data.copy()
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            
            # Remove rows with invalid dates
            data = data.dropna(subset=['Date'])
            
            # Quantity anomalies (using z-score method)
            quantity_mean = data['Quantity'].mean()
            quantity_std = data['Quantity'].std()
            quantity_threshold = 3  # Standard deviations
            
            quantity_anomalies = data[abs(data['Quantity'] - quantity_mean) > quantity_threshold * quantity_std]
            
            # Price anomalies
            price_mean = data['Price'].mean()
            price_std = data['Price'].std()
            price_threshold = 3
            
            price_anomalies = data[abs(data['Price'] - price_mean) > price_threshold * price_std]
            
            # Format date before storing in anomalies
            anomalies['Quantity'] = {
                'count': len(quantity_anomalies),
                'details': quantity_anomalies.nlargest(3, 'Quantity')[['Date', 'NDC', 'Quantity']].copy()
            }
            
            anomalies['Price'] = {
                'count': len(price_anomalies),
                'details': price_anomalies.nlargest(3, 'Price')[['Date', 'NDC', 'Price']].copy()
            }
            
            # Convert dates to string format for display
            anomalies['Quantity']['details']['Date'] = anomalies['Quantity']['details']['Date'].dt.strftime('%Y-%m-%d')
            anomalies['Price']['details']['Date'] = anomalies['Price']['details']['Date'].dt.strftime('%Y-%m-%d')
            
            # Detect sudden spikes (day-over-day changes)
            daily_data = data.groupby('Date').agg({
                'Quantity': 'sum',
                'Price': 'sum'
            }).reset_index()
            
            daily_changes = daily_data.set_index('Date').pct_change()
            spike_threshold = 0.5  # 50% change
            
            quantity_spikes = daily_changes[abs(daily_changes['Quantity']) > spike_threshold]
            price_spikes = daily_changes[abs(daily_changes['Price']) > spike_threshold]
            
            anomalies['Spikes'] = {
                'Quantity': len(quantity_spikes),
                'Price': len(price_spikes)
            }
            
            return anomalies
            
        except Exception as e:
            return {'error': str(e)}

    def _display_anomalies(self, anomalies, container):
        """Display anomalies in a formatted way"""
        with container:
            st.markdown("### Anomaly Detection")
            
            if 'error' in anomalies:
                st.warning(f"Could not detect anomalies: {anomalies['error']}")
                return
                
            # Display quantity anomalies
            st.markdown(f"**Unusual Quantities:** {anomalies['Quantity']['count']} detected")
            if anomalies['Quantity']['count'] > 0:
                st.markdown("Top 3 quantity anomalies:")
                st.dataframe(
                    anomalies['Quantity']['details'].style.format({
                        'Quantity': '{:,.0f}'
                    }),
                    use_container_width=True,
                    height=150
                )
            
            # Display price anomalies
            st.markdown(f"**Unusual Prices:** {anomalies['Price']['count']} detected")
            if anomalies['Price']['count'] > 0:
                st.markdown("Top 3 price anomalies:")
                st.dataframe(
                    anomalies['Price']['details'].style.format({
                        'Price': '${:,.2f}'
                    }),
                    use_container_width=True,
                    height=150
                )
            
            # Display spikes
            st.markdown("**Sudden Changes:**")
            st.markdown(f"- Quantity spikes: {anomalies['Spikes']['Quantity']}")
            st.markdown(f"- Price spikes: {anomalies['Spikes']['Price']}")

    def _show_analysis_page(self):
        """Show the analysis page"""
        st.title("Data Analysis")
        
        # Back button
        if st.button("← Back to Upload", type="secondary"):
            st.session_state.show_analysis = False
            st.rerun()
            return

        try:
            # Combine all mapped dataframes with year tracking
            all_data = []
            for year, data in st.session_state.uploaded_files.items():
                df = data['mapped_df'].copy()
                
                # Convert date with MMM YYYY format
                df['Date'] = pd.to_datetime(df['Date'].astype(str) + '-01', format='%b %Y-%d', errors='coerce')
                
                # Convert other columns - ensure Price is float
                df['NDC'] = df['NDC'].astype(str)
                df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
                df['Price'] = pd.to_numeric(df['Price'].replace('[\$,]', '', regex=True), errors='coerce')
                
                df['Year'] = year
                all_data.append(df)
            
            if not all_data:
                st.error("No data available for analysis. Please upload files first.")
                return
                
            all_data = pd.concat(all_data, ignore_index=True)
            
            # Remove rows with invalid data
            valid_mask = ~pd.isna(all_data[['Date', 'Quantity', 'Price']]).any(axis=1)
            all_data = all_data[valid_mask]
            
            if len(all_data) == 0:
                st.error("No valid data after conversion. Please check data format.")
                return
            
            # Calculate total value
            total_value = (all_data['Quantity'] * all_data['Price']).sum()
            
            # Overall summary section
            st.markdown("## Overall Data Summary")
            
            summary_stats = {
                "Total Records": f"{len(all_data):,}",
                "Date Range": f"{all_data['Date'].min().strftime('%b %Y')} to {all_data['Date'].max().strftime('%b %Y')}",
                "Total NDCs": f"{all_data['NDC'].nunique():,}",
                "Total Quantity": f"{all_data['Quantity'].sum():,.0f}",
                "Total Value": f"${total_value:,.2f}"
            }
            
            # Display overall summary statistics
            for label, value in summary_stats.items():
                st.markdown(f"**{label}:** {value}")

            # Analysis Selection Section
            st.markdown("## Select Analysis Type")
            
            # Create three columns for analysis boxes
            col1, col2, col3 = st.columns(3)
            
            with col1:
                abc_box = st.container(border=True)
                with abc_box:
                    st.markdown("### ABC Analysis")
                    st.markdown("Analyze inventory categorization")
                    if st.button("Run ABC Analysis", key="abc_button", use_container_width=True):
                        st.session_state.show_abc = True
                        st.session_state.show_trends = False
                        st.session_state.show_anomalies = False
                        st.rerun()

            with col2:
                trend_box = st.container(border=True)
                with trend_box:
                    st.markdown("### Trend Analysis")
                    st.markdown("Analyze patterns and trends")
                    if st.button("Run Trend Analysis", key="trend_button", use_container_width=True):
                        st.session_state.show_abc = False
                        st.session_state.show_trends = True
                        st.session_state.show_anomalies = False
                        st.rerun()

            with col3:
                anomaly_box = st.container(border=True)
                with anomaly_box:
                    st.markdown("### Anomaly Detection")
                    st.markdown("Detect unusual patterns")
                    if st.button("Run Anomaly Detection", key="anomaly_button", use_container_width=True):
                        st.session_state.show_abc = False
                        st.session_state.show_trends = False
                        st.session_state.show_anomalies = True
                        st.rerun()

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.markdown("Please ensure all data is properly formatted:")
            st.markdown("- Date should be in Month Year format (e.g., 'Oct 2023')")
            st.markdown("- Quantity should be numeric")
            st.markdown("- Price should be numeric (can include $ and commas)")

    def _show_trend_analysis(self):
        """Show trend analysis page"""
        st.title("Trend Analysis")
        
        if st.button("← Back to Analysis", type="secondary"):
            st.session_state.show_trends = False
            st.rerun()
            return
        
        # Add your existing trend analysis code here

    def _show_anomaly_detection(self):
        """Show anomaly detection page"""
        st.title("Anomaly Detection")
        
        if st.button("← Back to Analysis", type="secondary"):
            st.session_state.show_anomalies = False
            st.rerun()
            return
        
        # Add your existing anomaly detection code here

    def _show_abc_analysis(self):
        """Show ABC analysis page"""
        st.title("ABC Analysis")
        
        if st.button("← Back to Analysis", type="secondary"):
            st.session_state.show_abc = False
            st.rerun()
            return
        
        # Add your ABC analysis code here

if __name__ == "__main__":
    interface = DataUploadInterface()
    interface.create_interface()