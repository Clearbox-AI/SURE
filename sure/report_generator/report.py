import streamlit as st
import pandas as pd

from report_generator import _load_from_json, _convert_to_dataframe

# @st.cache_data
def display_feature_data(data):
    ''' This function displays the data for a selected feature
    '''
    # Get list of feature names
    feature_names = list(data.keys())

    # Create dropdown menu for feature selection
    selected_feature = st.selectbox('Select a feature]', [''] + feature_names)

    # If a feature has been selected, display its data and create another dropdown menu
    if selected_feature:
        # Get data for selected feature
        feature_data = data[selected_feature]

        # Convert data to DataFrame
        df_real = pd.DataFrame(feature_data['real'])
        df_real.index = ['real']
        df_synthetic = pd.DataFrame(feature_data['synthetic'])
        df_synthetic.index = ['synthetic']
        df = pd.concat([df_real, df_synthetic])

        # Display DataFrame
        st.write(df)

        # Remove selected feature from list of feature names
        feature_names.remove(selected_feature)

        # If there are still features left, create another dropdown menu
        if feature_names:
            st.write('Select another feature:')
            display_feature_data({name: data[name] for name in feature_names})

@st.cache_data
def ml_utility(obj):
    return _convert_to_dataframe(obj).set_index(['Model'])

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

    # Statistical similarity
    st.subheader("Statistical similarity")

    ### General statistics
    if "num_features_comparison" in st.session_state and st.session_state["num_features_comparison"]:
        display_feature_data(st.session_state["num_features_comparison"])
    if "cat_features_comparison" in st.session_state and st.session_state["cat_features_comparison"]:
        st.dataframe(st.session_state["cat_features_comparison"])
    if "time_features_comparison" in st.session_state and st.session_state["time_features_comparison"]:
        st.dataframe(st.session_state["time_features_comparison"])

    ### Correlation
    if "real_corr" in st.session_state:
        cb_rea_corr = st.checkbox("Show real dataset correlation matrix", value=False)
        if cb_rea_corr:
            st.dataframe(st.session_state["real_corr"])
    if "synth_corr" in st.session_state:
        cb_synth_corr = st.checkbox("Show synthetic dataset correlation matrix", value=False)
        if cb_synth_corr:
            st.dataframe(st.session_state["synth_corr"])
    if "diff_corr" in st.session_state:
        cb_diff_corr = st.checkbox("Show difference between the correlation matrix of the real dataset and the one of the synthetic dataset", value=False)
        if cb_diff_corr:
            st.dataframe(st.session_state["diff_corr"])
    
    # Utility 
    st.subheader("ML utility")
    if "models" in st.session_state:
        models_df = ml_utility(st.session_state["models"])
        st.dataframe(models_df.style.highlight_max(axis=0, subset=models_df.columns[:-1], color="#0D929A"))

if __name__ == "__main__":
    main()