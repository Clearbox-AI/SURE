import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.objects as so

@st.cache_data
def plot_DCR(train_data, val_data=None):
    """
    Plot histograms for Distance to Closest Record (DCR) for synthetic training and validation datasets.

    This function creates histograms to visualize the distribution of DCR values for synthetic training data.
    If validation data is provided, it also plots the DCR distribution for the synthetic validation data.

    Parameters
    ----------
    train_data : array-like
        DCR values for the synthetic training dataset.
    val_data : array-like, optional
        DCR values for the synthetic validation dataset. If None, only the synthetic training DCR histogram
        is plotted. Default is None.

    Returns
    -------
    None
        The function does not return any value. It plots the histograms using Streamlit.
    """
    # Convert data to pandas DataFrame
    df = pd.DataFrame({'DCR': train_data, 'Data': 'Synthetic-Trainining DCR'})
    if val_data is not None:
        df_val = pd.DataFrame({'DCR': val_data, 'Data': 'Synthetic-Validation DCR'})
        df = pd.concat([df, df_val])

    # colors = ['#6268ff','#ccccff']
    # chart = alt.Chart(df).mark_bar(opacity=0.6).encode(
    #     alt.X('DCR:Q', bin=alt.Bin(maxbins=15)),
    #     alt.Y('count()', stack=None),
    #     color=alt.Color('Data:N', scale=alt.Scale(range=colors))
    # ).properties(
    #     title='Histograms of Synthetic Train and Validation Data' if val_data is not None else 'Histograms of Synthetic Train',
    #     width=600,
    #     height=400
    # )
    # # Display chart in Streamlit
    # st.altair_chart(chart)

    f = plt.figure(figsize=(8, 4))
    sf = f.subfigures(1, 1)
    (
        so.Plot(df, x="DCR")
        .facet("Data")
        .add(so.Bars(color="#6268ff"), so.Hist())
        .on(sf)
        .plot()
    )

    for ax in sf.axes:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6)
        plt.setp(ax.get_yticklabels(), fontsize=8)
        ax.set_xlabel(ax.get_xlabel(), fontsize=6)  # Set x-axis label font size
        ax.set_ylabel(ax.get_ylabel(), fontsize=6)  # Set y-axis label font size
        ax.set_title(ax.get_title(), fontsize=8)    # Set title font size

    # Display the plot in Streamlit
    st.pyplot(f)

@st.cache_data
def dcr_stats_table(train_stats, val_stats=None):
    """
    Display a table with overall Distance to Closest Record (DCR) statistics.

    Parameters
    ----------
    train_stats : dict
        Dictionary containing statistics for the synthetic training dataset.
    val_stats : dict, optional
        Dictionary containing statistics for the synthetic validation dataset. Default is None.

    Returns
    -------
    None
        The function outputs a table using Streamlit.
    """
    df1 = pd.DataFrame.from_dict(train_stats, orient='index', columns=['Synth-Train'])
    if val_stats:
        df2 = pd.DataFrame.from_dict(val_stats, orient='index', columns=['Synth-Val'])
    # Merge the train and val DataFrames
    merged_df = pd.concat([df1, df2], axis=1)
    st.table(merged_df)

@st.cache_data
def dcr_validation(dcr_val, dcr_zero_train=None, dcr_zero_val=None):
    """
    Display the DCR share value and additional metrics for clones.

    Parameters
    ----------
    dcr_val : dict
        Dictionary containing DCR share values and warnings.
    dcr_zero_train : int, optional
        Number of clones in the synthetic training dataset. Default is None.
    dcr_zero_val : int, optional
        Number of clones in the synthetic validation dataset. Default is None.

    Returns
    -------
    None
        The function outputs metrics and information using Streamlit.
    """
    perc = dcr_val["percentage"]
    # st.write("The share of records of the synthetic dataset that are closer to the training set than to the validation set is: ", round(perc,1),"%")
    cols = st.columns(2)
    with cols[0]:
        st.metric("DCR closer to training", str(round(perc,1))+"%", help="For each synthetic record we computed its DCR with respect to the training set as well as with respect to the validation set. The validation set was not used to train the generative model. If we can find that the DCR values between synthetic and training (histogram on the left) are not systematically smaller than the DCR values between synthetic and validation (histogram on the right), we gain evidence of a high level of privacy.The share of records that are then closer to a training than to a validation record serves us as our proposed privacy risk measure. If that resulting share is then close to (or smaller than) 50%, we gain empirical evidence of the training and validation data being interchangeable with respect to the synthetic data. This in turn allows to make a strong case for plausible deniability for any individual, as the synthetic data records do not allow to conjecture whether an individual was or was not contained in the training dataset.")
    with cols[1]:
        if dcr_val["warnings"] != '':
            st.write(dcr_val["warnings"])
        # st.caption("N.B. the closer this value is to 50%, the less the synthetic dataset is vulnerable to reidentification attacks!")
    
    cols2 = st.columns(2)
    # with cols2[0]:
    if dcr_zero_train:
        # Table with the number of DCR equal to zero
        st.metric("Clones Synth-Train", dcr_zero_train, help="The number of clones shows how many rows of the synthetic dataset have an identical match in the training dataset. A very low value indicates a low risk in terms of privacy. Ideally this value should be close to 0, but some peculiar characteristics of the training dataset (small size or low column cardinality) may lead to a higher value. The duplicatestable shows the number of duplicates (identical rows) in the training dataset and the synthetic dataset: similar percentages mean higher utility.")
    # with cols2[1]:
    if dcr_zero_val:
        st.metric("Clones Synth-Val", dcr_zero_val, help="The number of clones shows how many rows of the synthetic dataset have an identical match in the validation dataset. A very low value indicates a low risk in terms of privacy. Ideally this value should be close to 0, but some peculiar characteristics of the training dataset (small size or low column cardinality) may lead to a higher value.")

def _MIA():
    cols = st.columns([1,3,2])
    with cols[0]:
        st.metric("MI mean risk score", str(round(st.session_state["MIA_attack"]["membership_inference_mean_risk_score"],3)*100)+"%", help="The MI Risk score is computed as (precision - 0.5) * 2.\n MI Risk Score smaller than 0.2 (20%) are considered to be very LOW RISK of disclosure due to membership inference.")
    with cols[1]:
        df_MIA = pd.DataFrame(st.session_state["MIA_attack"])
        st.dataframe(df_MIA.drop(columns=df_MIA.columns[-1]).iloc[::-1], hide_index=True)

def main():
    """
    Main function to configure and run the Streamlit application.

    The application provides a report on privacy metrics for a synthetic dataset,
    using the SURE library to visualize and interpret the data.

    Returns
    -------
    None
        The function runs the Streamlit app.
    """
    # Set app conifgurations
    st.set_page_config(layout="wide", page_title='SURE', page_icon=':large_purple_square:')

    # Header, subheader and description
    st.title('SURE')
    st.subheader('Synthetic Data: Utility, Regulatory compliance, and Ethical privacy')
    st.write(
        """This report provides a visual digest of the privacy metrics computed 
            with the library [SURE](https://github.com/Clearbox-AI/SURE) on the synthetic dataset under test.""")
    
    ### PRIVACY
    st.header("Privacy", divider='violet')
    st.sidebar.markdown("# Privacy")
    
    ## Distance to closest record
    st.subheader("Distance to closest record", help="Distances-to-Closest-Record are individual-level distances of synthetic records with respect to their corresponding nearest neighboring records from the training dataset. A DCR of 0 corresponds to an identical match. These histograms are used to assess whether the synthetic data is a simple copy or minor perturbation of the training data, resulting in high risk of disclosure. There is one DCR histogram computed only on the Training Set (histogram on the left) and one computed for the Synthetic Set vs Training Set (histogram on the right). Ideally, the two histograms should have a similar shape and, above all, the histogram on the right should be far enough away from 0.")
    # DCR statistics
    st.write("DCR statistics")
    col1, buff, col2 = st.columns([3,0.5,3])
    with col1:
        if "dcr_synth_train_stats" in st.session_state:
            dcr_stats_table(st.session_state["dcr_synth_train_stats"], 
                            st.session_state["dcr_synth_val_stats"])
    with col2:
        if "dcr_validation" in st.session_state:
            dcr_validation(st.session_state["dcr_validation"],
                            st.session_state["dcr_synth_train_num_of_zeros"], 
                            st.session_state["dcr_synth_val_num_of_zeros"])
        
    # Synth-train DCR and synth-validation DCR histograms
    if "dcr_synth_train" in st.session_state:
        plot_DCR(st.session_state["dcr_synth_train"], 
                    st.session_state["dcr_synth_val"])
    
    st.divider()

    ## Membership Inference Attack
    st.subheader("Membership Inference Attack", help="Membership inference attacks seek to infer membership of an individual record in the training set from which the synthetic data was generated. We consider a hypothetical adversary which has access to a subset of records K containing half instances from the training set and half instances from the validation set and that attempts a membership inference attack as follows: given an individual record k in K and the synthetic set S, the adversary identifies the closest record s in S; the adversary determines that k is part of the training set if d(k, s) is lower than a certain threshold. We evaluate the success rate of such attack strategy. Precision represents the number of correct decisions the adversary has made. Since 50% of the instances in K come from the training set and 50% come from the validation set, the baseline precision is 0.5, corresponding to a random choice and any value above that reflects an increasing levels of disclosure risk.")
    if "MIA_attack" in st.session_state:
        _MIA()        

if __name__ == "__main__":
    main()
