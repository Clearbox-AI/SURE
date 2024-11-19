[![PyPI](https://badge.fury.io/py/clearbox-sure.svg)](https://badge.fury.io/py/clearbox-sure)
[![Downloads](https://pepy.tech/badge/clearbox-sure)](https://pepy.tech/project/clearbox-sure)
[![GitHub stars](https://img.shields.io/github/stars/Clearbox-AI/SURE?style=social)](https://github.com/Clearbox-AI/SURE)

<p align="center">
    <a href="https://clearbox.ai"><img width="350px" src="./img/sure_logo.png" /></a><br />
    <i>Synthetic Data: Utility, Regulatory compliance, and Ethical privacy</i>
</p>

The SURE package is an open-source Python library for the assessment of the utility and privacy performance of any tabular synthetic dataset.

The SURE library works both with [pandas](https://pandas.pydata.org/) and [polars](https://pola.rs/) DataFrames.

# Installation

To install the library run the following command in your terminal:

```shell
$ pip install clearbox-sure
```

# Usage

The user must provide both the original real training dataset (which was used to train the generative model that produced the synthetic dataset), the real holdout dataset (which was NOT used to train the generative model that produced the synthetic dataset) and the corresponding synthetic dataset to enable the library's modules to perform the necessary computations for evaluation.

Follow the step-by-step guide to test the library using the provided [instructions](https://github.com/Clearbox-AI/SURE/blob/main/testing/sure_test.ipynb).

```python
# Import the necessary modules from the SURE library
from sure import Preprocessor, report
from sure.utility import (compute_statistical_metrics, compute_mutual_info,
			  compute_utility_metrics_class)
from sure.privacy import (distance_to_closest_record, dcr_stats, number_of_dcr_equal_to_zero, validation_dcr_test, 
			  adversary_dataset, membership_inference_test)

# Assuming real_data, valid_data and synth_data are three pandas DataFrames

# Real dataset - Preprocessor initialization and query exacution
preprocessor            = Preprocessor(real_data, get_discarded_info=False)
real_data_preprocessed  = preprocessor.transform(real_data, num_fill_null='forward', scaling='standardize')

# Validation dataset - Preprocessor initialization and query exacution
preprocessor            = Preprocessor(valid_data, get_discarded_info=False)
valid_data_preprocessed = preprocessor.transform(valid_data, num_fill_null='forward', scaling='standardize')

# Synthetic dataset - Preprocessor initialization and query exacution
preprocessor            = Preprocessor(synth_data, get_discarded_info=False)
synth_data_preprocessed = preprocessor.transform(synth_data, num_fill_null='forward', scaling='standardize')

# Statistical properties and mutual information
num_features_stats, cat_features_stats, temporal_feat_stats = compute_statistical_metrics(real_data_preprocessed, synth_data_preprocessed)
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
dcr_synth_train       = distance_to_closest_record("synth_train", synth_data_preprocessed, real_data_preprocessed)
dcr_synth_valid       = distance_to_closest_record("synth_val", synth_data_preprocessed, valid_data_preprocessed)
dcr_stats_synth_train = dcr_stats("synth_train", dcr_synth_train)
dcr_stats_synth_valid = dcr_stats("synth_val", dcr_synth_valid)
dcr_zero_synth_train  = number_of_dcr_equal_to_zero("synth_train", dcr_synth_train)
dcr_zero_synth_valid  = number_of_dcr_equal_to_zero("synth_val", dcr_synth_valid)
share                 = validation_dcr_test(dcr_synth_train, dcr_synth_valid)

# ML privacy attack sandbox initialization and simulation
adversary_dataset = adversary_dataset(real_data_preprocessed, valid_data_preprocessed)
# The function adversary_dataset adds a column "privacy_test_is_training" to the adversary dataset, indicating whether the record was part of the training set or not
adversary_guesses_ground_truth = adversary_dataset["privacy_test_is_training"] 
MIA = membership_inference_test(adversary_dataset, synth_data_preprocessed, adversary_guesses_ground_truth)

# Report generation as HTML page
report()
```
