import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Callable

class FileHandler:
    REQUIRED_COLUMNS = {
        'Date': 'Date (Month Year)',
        'NDC': 'NDC Number',
        'Quantity': 'Quantity',
        'Price': 'Price'
    }

    def __init__(self, on_proceed: Callable):
        self.on_proceed = on_proceed

    def show_upload_interface(self):
        """Show the file upload interface"""
        st.title("File Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload your Excel files",
            type=['xlsx', 'xls'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for file in uploaded_files:
                self._handle_file_upload(file)
            
            # Show proceed button only if files are mapped
            if any(st.session_state.uploaded_files):
                if st.button("Proceed to Analysis", type="primary"):
                    self.on_proceed()

    def _handle_file_upload(self, file):
        """Handle single file upload and mapping"""
        try:
            # Read Excel file
            df = pd.read_excel(file)
            
            # Store original DataFrame temporarily
            if 'temp_df' not in st.session_state:
                st.session_state.temp_df = {}
            st.session_state.temp_df[file.name] = df
            
            # Show original data preview
            st.markdown(f"### Original Data Preview: {file.name}")
            st.dataframe(df.head(), use_container_width=True)
            
            st.markdown("### Map Columns")
            
            # Get column mappings
            mapped_columns = self._get_column_mappings(df.columns, file.name)
            
            if mapped_columns:
                # Apply mappings and store processed dataframe
                mapped_df = df.rename(columns={v: k for k, v in mapped_columns.items()})
                required_cols = list(self.REQUIRED_COLUMNS.keys())
                
                # Verify all required columns are present
                if all(col in mapped_df.columns for col in required_cols):
                    st.markdown("### Preview of Mapped Data")
                    st.dataframe(mapped_df[required_cols].head(), use_container_width=True)
                    
                    st.session_state.uploaded_files[file.name] = {
                        'filename': file.name,
                        'mapped_df': mapped_df[required_cols]
                    }
                    st.success(f"Successfully mapped columns for {file.name}")
                else:
                    st.error("Please map all required columns")
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")

    def _get_column_mappings(self, available_columns: List[str], file_key: str) -> Dict[str, str]:
        """Get column mappings from user"""
        mappings = {}
        
        # Create a unique key for each file's mapping
        mapping_key = f"mapping_{file_key}"
        
        # Initialize session state for this file's mapping if not exists
        if mapping_key not in st.session_state:
            st.session_state[mapping_key] = {}
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Required Column**")
            for req_col, description in self.REQUIRED_COLUMNS.items():
                st.markdown(f"• {description}")
        
        with col2:
            st.markdown("**Select Matching Column**")
            # Create selectboxes for each required column
            for req_col, description in self.REQUIRED_COLUMNS.items():
                selected = st.selectbox(
                    f"Select column for {description}",
                    options=[''] + list(available_columns),
                    key=f"{mapping_key}_{req_col}"
                )
                if selected:
                    mappings[req_col] = selected
        
        # Add reset button
        if st.button("Reset Mapping", key=f"reset_{file_key}"):
            self._clear_mapping(file_key)
            st.rerun()
        
        return mappings

    def _clear_mapping(self, file_key: str):
        """Clear mapping for a specific file"""
        mapping_key = f"mapping_{file_key}"
        if mapping_key in st.session_state:
            del st.session_state[mapping_key]
        if file_key in st.session_state.uploaded_files:
            del st.session_state.uploaded_files[file_key]