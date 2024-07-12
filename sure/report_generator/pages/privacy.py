import streamlit as st
import plotly.figure_factory as ff
import altair as alt
import pandas as pd

@st.cache_data
def plot_DCR(train_data, val_data=None):
    ''' This function plots the synth-train DCR and synth-validation DCR histograms
    '''
    # Convert data to pandas DataFrame
    df = pd.DataFrame({'DCR': train_data, 'Data': 'Synthetic-Trainining DCR'})
    if val_data is not None:
        df_val = pd.DataFrame({'DCR': val_data, 'Data': 'Synthetic-Validation DCR'})
        df = pd.concat([df, df_val])

    # Create Altair chart
    chart = alt.Chart(df).mark_bar(opacity=0.6).encode(
        alt.X('DCR:Q', bin=alt.Bin(maxbins=10)),
        alt.Y('count()', stack=None),
        color='Data:N'
    ).properties(
        title='Histograms of Synthetic Train and Validation Data' if val_data is not None else 'Histograms of Synthetic Train',
        width=600,
        height=400
    )

    # Display chart in Streamlit
    st.altair_chart(chart)

@st.cache_data
def dcr_stats_table(train_stats, val_stats=None, dcr_zero_train=None, dcr_zero_val=None):
    ''' This function displays the table with the DCR overall stats
    '''
    df1 = pd.DataFrame.from_dict(train_stats, orient='index', columns=['Synth-Train'])
    if val_stats:
        df2 = pd.DataFrame.from_dict(val_stats, orient='index', columns=['Synth-Val'])
    # Merge the train and val DataFrames
    merged_df = pd.concat([df1, df2], axis=1)
    if dcr_zero_train:
        # Append the row with the number of DCR equal to zero
        new_row = pd.DataFrame({"Synth-Train":[dcr_zero_train], "Synth-Val":[dcr_zero_val]}, index=['DCR equal to zero'])
        merged_df = merged_df.append(new_row)
    st.table(merged_df)

@st.cache_data
def dcr_validation(dcr_val):
    perc = dcr_val["percentage"]
    st.write("The share of records of the synthetic dataset that are closer to the training set than to the validation set is: ",perc,"%")
    if dcr_val["warnings"] != '':
        st.write(dcr_val["warnings"])
    st.caption("N.B. the closer this value is to 50%, the less the synthetic dataset is vulnerable to reidentification attacks!")

def main():
    st.set_page_config(layout="wide")

    st.title('SURE')
    st.subheader('Synthetic Data: Utility, Regulatory compliance, and Ethical privacy')
    st.write(
        """This report provides a visual digest of the privacy metrics computed 
            with the library [SURE](https://github.com/Clearbox-AI/SURE) on the synthetic dataset under test.""")
    st.header("Privacy", divider='violet')
    st.sidebar.markdown("# Privacy")
    
    col1, buff, col2 = st.columns([3,0.5,3])
    with col1:
        st.write("DCR statistics")
        if "dcr_synth_train_stats" in st.session_state:
            dcr_stats_table(st.session_state["dcr_synth_train_stats"], 
                            st.session_state["dcr_synth_val_stats"], 
                            st.session_state["dcr_synth_train_num_of_zeros"], 
                            st.session_state["dcr_synth_val_num_of_zeros"])
        if "dcr_validation" in st.session_state:
            dcr_validation(st.session_state["dcr_validation"])
    with col2:
        # Synth-train DCR and synth-validation DCR histograms
        if "dcr_synth_train" in st.session_state:
            plot_DCR(st.session_state["dcr_synth_train"], 
                     st.session_state["dcr_synth_val"])

if __name__ == "__main__":
    main()
