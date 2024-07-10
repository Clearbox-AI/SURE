import streamlit as st
from report_generator import _load_from_json, _convert_to_dataframe

def main():
    st.title('SURE')
    st.subheader('Synthetic Data: Utility, Regulatory compliance, and Ethical privacy')

    data = _load_from_json()


if __name__ == "__main__":
    main()