import streamlit as st

def check_password():
    """Returns `True` if the user had the correct password."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        # First run, show input for password
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    else:
        # Password correct
        return True

def password_entered():
    """Checks whether a password entered by the user is correct."""
    if st.session_state["password"] == st.secrets["password"]:
        st.session_state["password_correct"] = True
        del st.session_state["password"]  # Don't store password
    else:
        st.session_state["password_correct"] = False

def main():
    st.title("🔐 Login")
    
    if check_password():
        st.switch_page("pages/1_Dashboard.py")  # Updated path to point to Dashboard in pages folder

if __name__ == "__main__":
    main() 