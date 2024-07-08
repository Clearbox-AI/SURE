# SURE 
### Synthetic Data: Utility, Regulatory compliance, and Ethical privacy

The SURE package is an open-source Python library intended to be used for the assessment of the utility and privacy performance of any tabular synthetic dataset.

The SURE library features multiple Python modules that can be easily imported and seamlessly integrated into any Python script after installing the library. 

### Modules overview

The SURE library features the following modules:

1. Preprocessor
2. Statistical similarity metrics
3. Model garden
4. ML utility metrics
5. Distance metrics
6. Privacy attack sandbox
7. Report generator

 **Preprocessor** 

The input datasets undergo manipulation by the preprocessor module, tailored to conform to the standard structure utilized across the subsequent processes. The Polars library used in the preprocessor makes this operation significantly faster compared to the use of other data processing libraries. 

 **Utility** 

The statistical similarity metrics, the ML utility metrics and the model garden modules constitute the data **utility evaluation** part.

The statistical similarity module and the distance metrics module take as input the pre-processed datasets and carry out the operation to assess the statistical similarity between the datasets and how different the content of the synthetic dataset is from the one of the original dataset.  In particular, The real and synthetic input datasets are used in the statistical similarity metrics module to assess how close the two datasets are in terms of statistical properties, such as mean, correlation, distribution.

The model garden executes a classification or regression task on the given dataset with multiple machine learning models, returning the performance metrics of each of the models tested on the given task and dataset.

The model garden module’s best performing models are employed in the machine learning utility metrics module to compute the usefulness of the synthetic data on a given ML task (classification or regression).

 **Privacy** 

The distance metrics and the privacy attack sandbox make up the synthetic data **privacy assessment** modules.

The distance metrics module computes the Gower distance between the two input datasets and the distance to the closest record for each line of the first dataset.

The ML privacy attack sandbox allows to simulate a Membership Inference Attack for re-identification of vulnerable records identified with the distance metrics module and evaluate how exposed the synthetic dataset is to this kind of assault.

 **Report** 

Eventually, the report generator provides a summary of the utility and privacy metrics computed in the previous modules, providing a visual digest with charts and tables of the results.

This following diagram serves as a visual representation of how each module contributes to the utility-privacy assessment process and highlights the seamless interconnection and synergy between individual blocks.

<img src="images/sure_workflow.png" alt="drawing" width="600"/>

SURE library workflow.

# Usage

The library leverages Polars, which ensures faster computations compared to other data manipulation libraries. It supports both Polars and Pandas dataframes.

The user must provide both the original real dataset and the corresponding synthetic dataset to enable the library's modules to perform the necessary computations for evaluation.

Below is a code snippet example for the usage of the library:

```python
# Import the necessary module from the SURE library
from sure import Preprocessor, statistical_similarity_metrics, distance_metrics, utility_metrics, Privacy_attack_sandobx, report_generator

# Real dataset - Preprocessor initialization and query exacution
preprocessor           = Preprocessor(real_data, get_discarded_info=False)
real_data_preprocessed = preprocessor.collect(real_data, num_fill_null='forward', scaling='standardize')

# Synthetic dataset - Preprocessor initialization and query exacution
preprocessor            = Preprocessor(synth_data, get_discarded_info=False)
synth_data_preprocessed = preprocessor.collect(synth_data, num_fill_null='forward', scaling='standardize')

~~~~# Compute the utility and privacy metrics on the real dataset and the synthetic dataset
util_metrics                  = utility_metrics(["Accuracy", "F1_score"], models="all", real_data_preprocessed, synth_data_preprocessed, labels)
simil_metrics                 = statistical_similarity_metrics(["simil_metric1", "simil_metric2"], real_data_preprocessed, synth_data_preprocessed)
DCR_train, DCR_holdout, share = distance_metrics(real_data_preprocessed, holdout_data_preprocessed, synth_data_preprocessed, dist_metric="Gower")

# ML privacy attack sanbox initialization and simulation
attack                      = Privacy_attack_sandobx("MIA", metrics="all")
attack_metrics, attack_pred = attack.simulate_attack(real_data_preprocessed, synth_data_preprocessed)

# Report generation as HTML page
report_generator(util_metrics, simil_metrics, DCR_train, DCR_holdout, share, attack_metrics, attack_pred)
```

Please review the dedicated documentation to learn how to further customize your synthetic data assessment pipeline.

# Installation

To install the library run the following command in your terminal:

```shell
$ pip install sure
$ sh install.sh
```