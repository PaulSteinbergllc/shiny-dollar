import streamlit as st

st.set_page_config(
    page_title="Dashboard",
    page_icon="🏠",
    layout="wide"
)

def create_clickable_box(title, description, page_name, emoji):
    box = st.container()
    with box:
        st.markdown(
            f"""
            <div style='
                padding: 20px;
                background-color: #1E1E1E;
                border-radius: 10px;
                text-align: center;
                cursor: pointer;
                height: 200px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                transition: transform 0.3s;
                color: white;
                &:hover {{
                    transform: scale(1.05);
                    background-color: #2E2E2E;
                }}
            '>
                <h1 style='font-size: 3em; margin-bottom: 10px;'>{emoji}</h1>
                <h2 style='color: white; margin-bottom: 10px;'>{title}</h2>
                <p style='color: #CCCCCC;'>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        # Make the entire container clickable
        if st.button("Click to navigate", key=title, help=description, use_container_width=True):
            try:
                st.switch_page(f"pages/{page_name}")
            except Exception as e:
                st.error(f"Could not navigate to {page_name}. Error: {str(e)}")

def main():
    st.title("🏠 Dashboard")
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create three columns for the boxes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_clickable_box(
            "Purchasing Data",
            "View and analyze purchasing data",
            "2_Purchasing.py",
            "📊"
        )
    
    with col2:
        create_clickable_box(
            "Receiving Data",
            "View and analyze receiving data",
            "3_Receiving.py",
            "📦"
        )
    
    with col3:
        create_clickable_box(
            "Waste Data",
            "View and analyze waste data",
            "4_Waste.py",
            "🗑️"
        )

if __name__ == "__main__":
    main() 