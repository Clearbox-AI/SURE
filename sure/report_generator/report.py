import streamlit as st

from report_generator import _load_from_json, _convert_to_dataframe

def main():
    st.set_page_config(layout="wide")

    # Header and subheader
    st.title('SURE')
    st.subheader('Synthetic Data: Utility, Regulatory compliance, and Ethical privacy')
    st.write(
        """This report provides a visual digest of the utility metrics computed 
            with the library [SURE](https://github.com/Clearbox-AI/SURE) on the synthetic dataset under test."""
    )
    st.header('Utility', divider='violet')
    st.sidebar.markdown("# Utility")
    
    # Load data in the session state, so that it is available in all the pages of the app
    st.session_state = _load_from_json()


if __name__ == "__main__":
    main()