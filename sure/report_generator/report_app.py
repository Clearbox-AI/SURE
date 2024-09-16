import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from report_generator import _load_from_json, _convert_to_dataframe

def _display_feature_data(data):
    ''' This function displays the data for a selected feature
    '''
    # Get list of feature names
    feature_names = list(data.keys())

    # Create dropdown menu for feature selection
    selected_feature = st.selectbox(label               = 'Select a statistical quantity:', 
                                    options             = ["Select a statistical quantity..."] + feature_names, 
                                    index               = None, 
                                    placeholder         = "Select a statistical quantity...", 
                                    label_visibility    = "collapsed")

    # If a feature has been selected, display its data and create another dropdown menu
    if selected_feature and selected_feature!="Select a statistical quantity...":
        # Get data for selected feature
        feature_data = data[selected_feature]

        # Convert data to DataFrame
        df_real              = pd.DataFrame(feature_data['real'], index=["Original"])
        df_synthetic         = pd.DataFrame(feature_data['synthetic'], index=["Synthetic"])
        df                   = pd.concat([df_real, df_synthetic])
        # Add row with the difference between the synthetic dataset and the real one
        df.loc['Difference'] = df.iloc[0] - df.iloc[1]

        # Display DataFrame
        st.write(df)

        # Remove selected feature from list of feature names
        feature_names.remove(selected_feature)

        # If there are still features left, create another dropdown menu
        if feature_names:
            _display_feature_data({name: data[name] for name in feature_names})

def _ml_utility():
    def _select_all():
        st.session_state["selected_models"] = models_df.index.values
    def _deselect_all():
        st.session_state["selected_models"] = []

    if 'selected_models' not in st.session_state:
        st.session_state["selected_models"] = []

    container = st.container()

    cols = st.columns([1.2,1])
    default = ["LinearSVC", "Perceptron", "LogisticRegression", "XGBClassifier", "SVC", "BaggingClassifier", "SGDClassifier", "RandomForestClassifier", "AdaBoostClassifier", "KNeighborsClassifier", "DecisionTreeClassifier", "DummyClassifier"]
    
    models_real_df = _convert_to_dataframe(st.session_state["models"]).set_index(['Model'])
    models_synth_df = _convert_to_dataframe(st.session_state["models_synth"]).set_index(['Model'])
    models_df = pd.concat([models_real_df, models_synth_df], axis=1)
    interleaved_columns = [col for pair in zip(models_real_df.columns, models_synth_df.columns) for col in pair]
    models_df = models_df[interleaved_columns]    
    models_delta_df = _convert_to_dataframe(st.session_state["models_delta"]).set_index(['Model'])
    
    st.session_state["selected_models"] = default
    
    options = st.multiselect(label="Select ML models to show in the table:", 
                    options=models_df.index.values, 
                    default= [x for x in st.session_state["selected_models"] if x in models_df.index.values], 
                    placeholder="Select ML models...",
                    key="models_multiselect")
    subcols = st.columns([1,1,3])
    with subcols[0]:    
        butt1 = st.button("Select all models")
    with subcols[1]:
        butt2 = st.button("Deselect all models")
    if butt1:
        options = models_df.index.values
    if butt2:
        options = []

    st.text("") # vertical space
    st.text("") # vertical space
    st.text("ML Metrics")
    st.dataframe(models_df.loc[options].style.highlight_max(axis=0, subset=models_df.columns[:-2], color="#5cbd91"))
    st.text("") # vertical space
    st.text("") # vertical space
    st.text("ML Delta Metrics", help="Difference between the metrics of the original dataset and the synthetic dataset.")
    st.dataframe(models_delta_df.abs().loc[options].style.highlight_min(axis=0, subset=models_delta_df.columns[:-1], color="#5cbd91").highlight_max(axis=0, subset=models_delta_df.columns[:-1], color="#c45454"))

# def _ml_utility(models_df):
#     def _select_all():
#         st.session_state['selected_models'] = models_df.index.values
#     def _deselect_all():
#         st.session_state['selected_models'] = []

#     if 'selected_models' not in st.session_state:
#         st.session_state['selected_models'] = ["LinearSVC", "Perceptron", "LogisticRegression", "XGBClassifier", "SVC", "BaggingClassifier", "SGDClassifier", "RandomForestClassifier", "AdaBoostClassifier", "KNeighborsClassifier", "DecisionTreeClassifier", "DummyClassifier"]
    
#     container = st.container()

#     col1, col2, col3 = st.columns([1, 1, 1])
#     with col1:
#         st.button("Select all models", on_click=_select_all, key="butt1")
#     with col2:
#         st.button("Deselect all models", on_click=_deselect_all, key="butt2")

#     selected_models = container.multiselect(
#         "Select ML models to show in the table:",
#         models_df.index.values,
#         default=st.session_state['selected_models'],
#         key='selected_models'
#         )   
    
#     with col1:
#         st.dataframe(models_df.loc[selected_models].style.highlight_max(axis=0, subset=models_df.columns[:-1], color="#99ffcc"))
#     with col2:
#         st.session_state['selected_models']
#     with col3:
#         selected_models

# def _ml_utility():
#     def select_all():
#         st.session_state['selected_models'] = ['A', 'B', 'C', 'D']
#     def deselect_all():
#         st.session_state['selected_models'] = []

#     if 'selected_models' not in st.session_state:
#         st.session_state['selected_models'] = []

#     container = st.container()

#     col1, col2 = st.columns([1, 1])
#     with col1:
#         st.button("Select all", on_click=select_all, key="butt1")
#     with col2:
#         st.button("Deselect all", on_click=deselect_all, key="butt2")

#     selected_models = container.multiselect(
#         "Select one or more options:",
#         ['A', 'B', 'C', 'D'],
#         default=st.session_state['selected_models'],
#         key='selected_models'
#     )
    
#     st.markdown('##')
#     cols = st.columns(2)
#     with cols[0]:
#         "st.session_state['selected_models']:"
#         st.session_state['selected_models']
#     with cols[1]:
#         "selected_models:"
#         selected_models

def plot_heatmap(data, title):
    df = pd.DataFrame(data)
    df = df.drop(columns=["Columns"])
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(df, dtype=bool), 1)

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(13, 11))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(20, 265, as_cmap=True)

    # Draw the heatmap with the mask, correct aspect ratio, and column names as labels
    sns.heatmap(df, cmap=cmap, center=0, 
                square=True, mask=mask, linewidths=.5, 
                cbar_kws={"shrink": .75},
                xticklabels=df.columns,  # Display column names on x-axis
                yticklabels=df.columns   # Display column names on y-axis
                )

    # Rotate x-axis labels for better readability and add title
    plt.xticks(rotation=45, ha='right')
    ax.set_title(title, fontsize=16, pad=20)

    # Display the plot
    st.pyplot(f)

def main(path_to_json):
    # Set app conifgurations
    st.set_page_config(layout="wide", page_title='SURE', page_icon=':large_purple_square:')
    
    # Header and subheader and description
    st.title('SURE')
    st.subheader('Synthetic Data: Utility, Regulatory compliance, and Ethical privacy')
    st.write(
        """This report provides a visual digest of the utility metrics computed 
            with the library [SURE](https://github.com/Clearbox-AI/SURE) on the synthetic dataset under test.""")
    
    ### UTILITY
    st.header('Utility', divider='violet')
    st.sidebar.markdown("# Utility")

    # Load data in the session state, so that it is available in all the pages of the app
    if path_to_json:
        st.session_state = _load_from_json(path_to_json)
    else:
        st.session_state = _load_from_json("")
    
    ## Statistical similarity
    st.subheader("Statistical similarity")

    # Features distribution
    # plot_distribution()

    # General statistics
    if "num_features_comparison" in st.session_state and st.session_state["num_features_comparison"]:
        features_comparison = st.session_state["num_features_comparison"]
    if "cat_features_comparison" in st.session_state and st.session_state["cat_features_comparison"]:
        if "features_comparison" in locals():
            features_comparison = {**features_comparison, **st.session_state["cat_features_comparison"]}
        else:
            features_comparison = st.session_state["cat_features_comparison"]
    if "time_features_comparison" in st.session_state and st.session_state["time_features_comparison"]:
        if "features_comparison" in locals():
            features_comparison = {**features_comparison, **st.session_state["time_features_comparison"]}
        else:
            features_comparison = st.session_state["time_features_comparison"]
    if features_comparison:
        _display_feature_data(features_comparison)

    st.markdown('###')

    # Correlation
    st.subheader("Feature correlation", help="This matrix shows the correlation between ordinal and categorical features. These correlation coefficients are obtained using the mutual information metric. Mutual information describes relationships in terms of uncertainty.")
    if "real_corr" in st.session_state:
        cb_rea_corr = st.checkbox("Show original dataset correlation matrix", value=False)
        if cb_rea_corr:
            plot_heatmap(st.session_state["real_corr"], 'Original Dataset Correlation Matrix Heatmap')
    if "synth_corr" in st.session_state:
        cb_synth_corr = st.checkbox("Show synthetic dataset correlation matrix", value=False)
        if cb_synth_corr:
            plot_heatmap(st.session_state["synth_corr"], 'Synthetic Dataset Correlation Matrix Heatmap')
    if "diff_corr" in st.session_state:
        cb_diff_corr = st.checkbox("Show difference between the correlation matrix of the original dataset and the one of the synthetic dataset", value=False)
        if cb_diff_corr:
            plot_heatmap(st.session_state["diff_corr"], 'Original-Synthetic Dataset Correlation Matrix Difference Heatmap')

    st.divider()

    ## ML Utility 
    st.subheader("ML utility")
    if "models" in st.session_state:
        _ml_utility()
        
if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="This script runs the utility and privacy report app of the SURE library.")
    parser.add_argument('path', type=str, default="", help='path where the json file with the results is saved')
    args = parser.parse_args()

    main(args.path)