import pandas as pd
import streamlit as st
import numpy as np
from typing import Dict, Any

class DataProcessor:
    @staticmethod
    def format_ndc(ndc: str) -> str:
        """Format NDC to 5-4-2 format"""
        try:
            # Remove any non-numeric characters
            ndc = ''.join(filter(str.isdigit, str(ndc)))
            
            # Pad with zeros if necessary to get 11 digits
            ndc = ndc.zfill(11)
            
            # Format as 5-4-2
            return f"{ndc[:5]}-{ndc[5:9]}-{ndc[9:]}"
        except Exception as e:
            st.error(f"Error formatting NDC {ndc}: {str(e)}")
            return ndc

    @staticmethod
    def process_uploaded_files():
        """Process all uploaded files and return combined DataFrame"""
        try:
            all_data = []
            
            if 'uploaded_files' not in st.session_state:
                st.error("No files uploaded")
                return None
                
            for file_name, data in st.session_state.uploaded_files.items():
                if 'mapped_df' not in data:
                    continue
                    
                df = DataProcessor._process_single_file(data['mapped_df'])
                if df is not None:
                    all_data.append(df)
            
            if not all_data:
                st.error("No data available for analysis")
                return None
                
            return pd.concat(all_data, ignore_index=True)
            
        except Exception as e:
            st.error(f"Error in process_uploaded_files: {str(e)}")
            return None

    @staticmethod
    def _process_single_file(df):
        """Process a single DataFrame"""
        try:
            if df is None:
                return None
                
            df = df.copy()
            
            # Convert date
            df['Date'] = pd.to_datetime(df['Date'].astype(str) + '-01', 
                                      format='%b %Y-%d', 
                                      errors='coerce')
            
            # Convert and format NDC
            df['NDC'] = df['NDC'].astype(str).apply(DataProcessor.format_ndc)
            
            # Convert other columns
            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
            df['Price'] = pd.to_numeric(df['Price'].replace('[\$,]', '', regex=True), 
                                      errors='coerce')
            
            # Add year from date
            df['Year'] = df['Date'].dt.year
            
            # Remove invalid rows
            valid_mask = ~pd.isna(df[['Date', 'Quantity', 'Price']]).any(axis=1)
            return df[valid_mask]
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return None