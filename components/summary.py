import streamlit as st

class SummaryStats:
    def display(self, data):
        """Display summary statistics"""
        st.markdown("## Overall Data Summary")
        
        total_value = (data['Quantity'] * data['Price']).sum()
        
        summary_stats = {
            "Total Records": f"{len(data):,}",
            "Date Range": f"{data['Date'].min().strftime('%b %Y')} to {data['Date'].max().strftime('%b %Y')}",
            "Total NDCs": f"{data['NDC'].nunique():,}",
            "Total Quantity": f"{data['Quantity'].sum():,.0f}",
            "Total Value": f"${total_value:,.2f}"
        }
        
        for label, value in summary_stats.items():
            st.markdown(f"**{label}:** {value}") 