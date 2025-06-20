<a href="https://dario-brunelli-clearbox-ai.notion.site/SURE-Documentation-2c17db370641488a8db5bce406032c1f"><img src="https://img.shields.io/badge/SURE-docs-blue?logo=mdbook" /></a>
[![Documentation Status](https://readthedocs.org/projects/clearbox-sure/badge/?version=latest)](https://clearbox-sure.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://badge.fury.io/py/clearbox-sure.svg)](https://badge.fury.io/py/clearbox-sure)
[![Downloads](https://pepy.tech/badge/clearbox-sure)](https://pepy.tech/project/clearbox-sure)
[![GitHub stars](https://img.shields.io/github/stars/Clearbox-AI/SURE?style=social)](https://github.com/Clearbox-AI/SURE)

<img src="docs/source/img/sure_logo_nobg.png" width="250">

### Synthetic Data: Utility, Regulatory compliance, and Ethical privacy

The SURE package is an open-source Python library intended to be used for the assessment of the utility and privacy performance of any tabular synthetic dataset.

The SURE library features multiple Python modules that can be easily imported and seamlessly integrated into any Python script after installing the library.

> [!WARNING]
> This is a beta version of the library and only runs on Linux and MacOS for the moment.

> [!IMPORTANT]
> Requires Python >= 3.10

# Installation

To install the library run the following command in your terminal:

```shell
$ pip install clearbox-sure
```

# Modules overview

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

The statistical similarity module and the distance metrics module take as input the pre-processed datasets and carry out the operation to assess the statistical similarity between the datasets and how different the content of the synthetic dataset is from the one of the original dataset. In particular, The real and synthetic input datasets are used in the statistical similarity metrics module to assess how close the two datasets are in terms of statistical properties, such as mean, correlation, distribution.

The model garden executes a classification or regression task on the given dataset with multiple machine learning models, returning the performance metrics of each of the models tested on the given task and dataset.

The model garden module’s best performing models are employed in the machine learning utility metrics module to compute the usefulness of the synthetic data on a given ML task (classification or regression).

**Privacy**

The distance metrics and the privacy attack sandbox make up the synthetic data **privacy assessment** modules.

The distance metrics module computes the Gower distance between the two input datasets and the distance to the closest record for each line of the first dataset.

The ML privacy attack sandbox allows to simulate a Membership Inference Attack for re-identification of vulnerable records identified with the distance metrics module and evaluate how exposed the synthetic dataset is to this kind of assault.

**Report**

Eventually, the report generator provides a summary of the utility and privacy metrics computed in the previous modules, providing a visual digest with charts and tables of the results.

This following diagram serves as a visual representation of how each module contributes to the utility-privacy assessment process and highlights the seamless interconnection and synergy between individual blocks.

<img src="docs/source/img/SURE_workflow_.png" alt="drawing" width="500"/>

# Usage

The library leverages Polars, which ensures faster computations compared to other data manipulation libraries. It supports both Polars and Pandas dataframes.

The user must provide both the original real training dataset (which was used to train the generative model that produced the synthetic dataset), the real holdout dataset (which was NOT used to train the generative model that produced the synthetic dataset) and the corresponding synthetic dataset to enable the library's modules to perform the necessary computations for evaluation.

Below is a code snippet example for the usage of the library:

```python
# Import the necessary modules from the SURE library
from sure import Preprocessor, report
from sure.utility import (compute_statistical_metrics, compute_mutual_info,
			  compute_utility_metrics_class,
			  detection,
			  query_power)
from sure.privacy import (distance_to_closest_record, dcr_stats, number_of_dcr_equal_to_zero, validation_dcr_test, 
			  adversary_dataset, membership_inference_test)

# Assuming real_data, valid_data and synth_data are three pandas DataFrames

# Preprocessor initialization and query execution on the real, synthetic and validation datasets
preprocessor            = Preprocessor(real_data, num_fill_null='forward', scaling='standardize')

real_data_preprocessed  = preprocessor.transform(real_data)
valid_data_preprocessed = preprocessor.transform(valid_data)
synth_data_preprocessed = preprocessor.transform(synth_data)

# Statistical properties and mutual information
num_features_stats, cat_features_stats, temporal_feat_stats = compute_statistical_metrics(real_data, synth_data)
corr_real, corr_synth, corr_difference                      = compute_mutual_info(real_data_preprocessed, synth_data_preprocessed)

# ML utility: TSTR - Train on Synthetic, Test on Real
X_train      = real_data_preprocessed.drop("label", axis=1) # Assuming the datasets have a “label” column for the machine learning task they are intended for
y_train      = real_data_preprocessed["label"]
X_synth      = synth_data_preprocessed.drop("label", axis=1)
y_synth      = synth_data_preprocessed["label"]
X_test       = valid_data_preprocessed.drop("label", axis=1).limit(10000) # Test the trained models on a portion of the original real dataset (first 10k rows)
y_test       = valid_data_preprocessed["label"].limit(10000)
TSTR_metrics = compute_utility_metrics_class(X_train, X_synth, X_test, y_train, y_synth, y_test)

# Distance to closest record
dcr_synth_train       = distance_to_closest_record("synth_train", synth_data, real_data)
dcr_synth_valid       = distance_to_closest_record("synth_val", synth_data, valid_data)
dcr_stats_synth_train = dcr_stats("synth_train", dcr_synth_train)
dcr_stats_synth_valid = dcr_stats("synth_val", dcr_synth_valid)
dcr_zero_synth_train  = number_of_dcr_equal_to_zero("synth_train", dcr_synth_train)
dcr_zero_synth_valid  = number_of_dcr_equal_to_zero("synth_val", dcr_synth_valid)
share                 = validation_dcr_test(dcr_synth_train, dcr_synth_valid)

# Detection Score
detection_score = detection(real_data, synth_data, preprocessor)

# Query Power
query_power_score = query_power(real_data, synth_data, preprocessor)

# ML privacy attack sandbox initialization and simulation
adversary_df = adversary_dataset(real_data, valid_data)
# The function adversary_dataset adds a column "privacy_test_is_training" to the adversary dataset, indicating whether the record was part of the training set or not
adversary_guesses_ground_truth = adversary_df["privacy_test_is_training"] 
MIA = membership_inference_test(adversary_df, synth_data, adversary_guesses_ground_truth)

# Report generation as HTML page
report(real_data, synth_data)
```

Follow the step-by-step [guide](https://github.com/Clearbox-AI/SURE/tree/main/examples) to test the library.

<!-- Review the dedicated [documentation](https://dario-brunelli-clearbox-ai.notion.site/SURE-Documentation-2c17db370641488a8db5bce406032c1f) to learn how to further customize your synthetic data assessment pipeline. -->
