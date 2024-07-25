import numpy as np
import pandas as pd
import polars as pl

from typing import Dict, List, Tuple

from sklearn.metrics import precision_score

from sure.privacy import distance_to_closest_record
from sure import _save_to_json, _drop_real_cols

# ATTACK SANDBOX
def _polars_to_pandas(dataframe: pl.DataFrame | pl.LazyFrame):
    if isinstance(dataframe, pl.DataFrame):
        dataframe = dataframe.to_pandas()
    if isinstance(dataframe, pl.LazyFrame):
        dataframe = dataframe.collect().to_pandas()
    return dataframe

def _pl_pd_to_numpy(dataframe: pl.DataFrame | pl.LazyFrame | pl.Series | pd.DataFrame):
    if isinstance(dataframe, (pl.DataFrame, pl.Series, pd.DataFrame)):
        dataframe = dataframe.to_numpy()
    if isinstance(dataframe, pl.LazyFrame):
        dataframe = dataframe.collect().to_numpy()
    return dataframe

def adversary_dataset(
    training_set: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    validation_set: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    original_dataset_sample_fraction: float = 0.2,
) -> pd.DataFrame:
    """
    Create an adversary dataset for the Membership Inference Test given a training
    and validation set. The validation set must be smaller than the training set.

    The size of the resulting adversary dataset is a fraction of the sum of the training
    set size and the validation set size.

    It takes half of the final rows from the training set and the other half from the
    validation set. It adds a column to mark which rows was sampled from the training set.

    Parameters
    ----------
    training_set : pd.DataFrame
        The training set as a pandas DataFrame.
    validation_set : pd.DataFrame
        The validation set as a pandas DataFrame.
    original_dataset_sample_fraction : float, optional
        How many rows (a fraction from 0  to 1) to sample from the concatenation of the
        training and validation set, by default 0.2

    Returns
    -------
    pd.DataFrame
        A new pandas DataFrame in which half of the rows come from the training set and
        the other half come from the validation set.
    """
    training_set = _polars_to_pandas(training_set)
    validation_set = _polars_to_pandas(validation_set)
    
    sample_number_of_rows = (
        training_set.shape[0] + validation_set.shape[0]
    ) * original_dataset_sample_fraction

    # if the validation set is very small, we'll set the number of rows to sample equal to
    # the number of rows of the validation set, that is every row of the validation set
    # is going into the adversary set.
    sample_number_of_rows = min(int(sample_number_of_rows / 2), validation_set.shape[0])

    sampled_from_training = training_set.sample(
        sample_number_of_rows, replace=False, random_state=42
    )
    sampled_from_training["privacy_test_is_training"] = True

    sampled_from_validation = validation_set.sample(
        sample_number_of_rows, replace=False, random_state=42
    )
    sampled_from_validation["privacy_test_is_training"] = False

    adversary_dataset = pd.concat(
        [sampled_from_training, sampled_from_validation], ignore_index=True
    )
    adversary_dataset = adversary_dataset.sample(frac=1).reset_index(drop=True)
    return adversary_dataset

def membership_inference_test(
    adversary_dataset:  pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    synthetic_dataset:  pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    adversary_guesses_ground_truth: np.ndarray | pd.DataFrame | pl.DataFrame | pl.LazyFrame | pl.Series,
    parallel: bool = True,
):
    ''' Simulate a Membership Inference Attack on the synthetic dataset provided, given an adversary dataset
    '''    
    # Convert datasets
    adversary_dataset     = _polars_to_pandas(adversary_dataset)
    synthetic_dataset     = _polars_to_pandas(synthetic_dataset)
    adversary_guesses_ground_truth  = _pl_pd_to_numpy(adversary_guesses_ground_truth)

    adversary_dataset=adversary_dataset.drop(["privacy_test_is_training"],axis=1)
    
    # Drop columns that are present in Y but are missing in X
    adversary_dataset = _drop_real_cols(synthetic_dataset, adversary_dataset)

    dcr_adversary_synth = distance_to_closest_record("other",
                                                adversary_dataset,
                                                synthetic_dataset,
                                                parallel=parallel,
                                                save_output=False
                                            )

    adversary_precisions = []
    distance_thresholds = np.quantile(
        dcr_adversary_synth, [0.5, 0.25, 0.2, np.min(dcr_adversary_synth) + 0.01]
    )
    for distance_threshold in distance_thresholds:
        adversary_guesses = dcr_adversary_synth < distance_threshold
        adversary_precision = precision_score(
            adversary_guesses_ground_truth, adversary_guesses, zero_division=0
        )
        adversary_precisions.append(max(adversary_precision, 0.5))
    adversary_precision_mean = np.mean(adversary_precisions).item()
    membership_inference_mean_risk_score = max(
        (adversary_precision_mean - 0.5) * 2, 0.0
    )

    attack_output = {
        "adversary_distance_thresholds": distance_thresholds.tolist(),
        "adversary_precisions": adversary_precisions,
        "membership_inference_mean_risk_score": membership_inference_mean_risk_score,
    }

    _save_to_json("MIA_attack", attack_output)
    return attack_output