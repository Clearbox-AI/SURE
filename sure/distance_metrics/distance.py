import pyximport
import os
from multiprocessing import Pool

import numpy  as np
import pandas as pd
import polars as pl
import polars.selectors as cs

from typing import Dict, List, Tuple, Union
int_type    = Union[int, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]
float_type  = Union[float, np.float16, np.float32, np.float64]

from ..report_generator.report_generator import _save_to_json

from sure import _drop_real_cols

pyximport.install(setup_args={"include_dirs": np.get_include()})
from sure.distance_metrics.gower_matrix_c import gower_matrix_c


def _polars_to_pandas(dataframe: pl.DataFrame | pl.LazyFrame):
    if isinstance(dataframe, pl.DataFrame):
        dataframe = dataframe.to_pandas()
    if isinstance(dataframe, pl.LazyFrame):
        dataframe = dataframe.collect().to_pandas()
    return dataframe

def _gower_matrix(
                X_categorical: np.ndarray,
                X_numerical: np.ndarray,
                Y_categorical: np.ndarray,
                Y_numerical: np.ndarray,
                numericals_ranges: np.ndarray,
                features_weight_sum: float,
                fill_diagonal: bool,
                first_index: int = -1,
            ) -> np.ndarray:
    """
    _summary_

    Parameters
    ----------
    X_categorical : np.ndarray
        2D array containing only the categorical features of the X dataframe as uint8 values, shape (x_rows, cat_features).
    X_numerical : np.ndarray
        2D array containing only the numerical features of the X dataframe as float32 values, shape (x_rows, num_features).
    Y_categorical : np.ndarray
        2D array containing only the categorical features of the Y dataframe as uint8 values, shape (y_rows, cat_features).
    Y_numerical : np.ndarray
        2D array containing only the numerical features of the Y dataframe as float32 values, shape (y_rows, num_features).
    numericals_ranges : np.ndarray
        1D array containing the range (max-min) of each numerical feature as float32 values, shap (num_features,).
    features_weight_sum : float
        Sum of the feature weights used for the final average computation (usually it's just the number of features, each
        feature has a weigth of 1).
    fill_diagonal : bool
       Whether to fill the matrix diagonal values with a value larger than 1 (5.0). It must be True to get correct values
       if your computing the matrix just for one dataset (comparing a dataset with itself), otherwise you will get DCR==0
       for each row because on the diagonal you will compare a pair of identical instances.
    first_index : int, optioanl
        This is required only in case of parallel computation: the computation will occur batch by batch so ther original
        diagonal values will no longer be on the diagonal on each batch. We use this index to fill correctly the diagonal
        values. If -1 it's assumed there's no parallel computation, by default -1

    Returns
    -------
    np.ndarray
        1D array containing the Distance to the Closest Record for each row of x_dataframe shape (x_dataframe rows, )
    """
    return gower_matrix_c(
        X_categorical,
        X_numerical,
        Y_categorical,
        Y_numerical,
        numericals_ranges,
        features_weight_sum,
        fill_diagonal,
        first_index,
    )

def distance_to_closest_record(
                        dcr_name: str,
                        x_dataframe: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
                        y_dataframe: pd.DataFrame | pl.DataFrame | pl.LazyFrame = None,
                        feature_weights: np.ndarray | List = None,
                        parallel: bool = True,
                        save_output: bool = True
                    ) -> np.ndarray:
    """
    Compute the distancees to closest record of dataframe X from dataframe Y using 
    a modified version of the Gower's distance. 
    The two dataframes may contain mixed datatypes (numerical and categorical).

    Paper references:
    * A General Coefficient of Similarity and Some of Its Properties, J. C. Gower
    * Dimensionality Invariant Similarity Measure, Ahmad Basheer Hassanat

    Parameters
    ----------
    dcr_name : str
        Name with which the DCR will be saved with in the JSON file used to generate the final report.
        Can be one of the following:
            - synth_train
            - synth_val
            - other
    x_dataframe : pd.DataFrame
        A dataset containing numerical and categorical data.
    categorical_features : List
        List of booleans that indicates which features are categorical.
        If categoricals_features[i] is True, feature i is categorical.
        Must have same length of x_dataframe.columns.
    y_dataframe : pd.DataFrame, optional
        Another dataset containing numerical and categorical data, by default None.
        It must contains the same columns of x_dataframe.
        If None, the distance matrix is computed between x_dataframe and x_dataframe
    feature_weights : List, optional
        List of features weights to use computing distances, by default None.
        If None, each feature weight is 1.0
    parallel : Boolean, optional
        Whether to enable the parallelization to compute Gower matrix, by default True
    save_output : bool
        If True, saves the DCR information into the JSON file used to generate the final report.

    Returns
    -------
    np.ndarray
        1D array containing the Distance to the Closest Record for each row of x_dataframe
        shape (x_dataframe rows, )

    Raises
    ------
    TypeError
        If dc_name is not one of the names listed above.
    TypeError
        If X and Y don't have the same (number of) columns.
    """
    if dcr_name != "synth_train" and dcr_name != "synth_val" and dcr_name != "other":
        raise TypeError("dcr_name must be one of the following:\n    -\"synth_train\"\n    -\"synth_val\"\n    -\"other\"")

    # Converting X Dataset in to pd.DataFrame
    X = _polars_to_pandas(x_dataframe)
      
    # Convert any temporal features to int
    temporal_columns = X.select_dtypes(include=['datetime']).columns
    X[temporal_columns] = X[temporal_columns].astype('int64')
  
    # se c'è un secondo dataframe le distanze vengono calcolate con esso, altrimente X con sè stesso
    if y_dataframe is None:
        Y = X
        fill_diagonal = True
    else:
        # Converting X Dataset in to pd.DataFrame
        Y = _polars_to_pandas(y_dataframe)
        fill_diagonal = False
        Y[temporal_columns] = Y[temporal_columns].astype('int64')

    # Drop columns that are present in Y but are missing in X
    Y = _drop_real_cols(X, Y)
    
    if not isinstance(X, np.ndarray):
        if not np.array_equal(X.columns, Y.columns):
            raise TypeError("X and Y dataframes have different columns.")
    else:
        if not X.shape[1] == Y.shape[1]:
            raise TypeError("X and Y arrays have different number of columns.")
        
    # Get categorical features
    categorical_features = np.array(X.dtypes)==pl.Utf8
    if not isinstance(categorical_features, np.ndarray):
        categorical_features = np.array(categorical_features)

    # Entrambi i dataframe vengono trasformati in array/matrice numpy
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)

    if feature_weights is None:
        # se non ho passato pesi specifici, tutti i pesi sono 1
        feature_weights = np.ones(X.shape[1])
    else:
        feature_weights = np.array(feature_weights)

    # La somma dei pesi è necessaria per fare la media alla fine (divisione)
    weight_sum = feature_weights.sum().astype("float32")

    # Matrice delle feature categoriche di X (num_rows_X x num_cat_feat)
    X_categorical = X[:, categorical_features].astype("uint8")

    # Matrice delle feature numeriche di X (num_rows_X x num_num_feat)
    X_numerical = X[:, np.logical_not(categorical_features)].astype("float32")

    # Matrice delle feature categoriche di Y (num_rows_Y x num_cat_feat)
    Y_categorical = Y[:, categorical_features].astype("uint8")

    # Matrice delle feature numeriche di Y (num_rows_Y x num_num_feat)
    Y_numerical = Y[:, np.logical_not(categorical_features)].astype("float32")

    # Il rango delle numeriche è necessario per il modo in cui sono calcolate le distanze Gower
    # Trovo il minimo e il massimo per ogni numerica concatenando X e Y quindi il rango sottraendo
    # tutti i minimi dai massimi.
    numericals_mins = np.amin(np.concatenate((X_numerical, Y_numerical)), axis=0)
    numericals_maxs = np.amax(np.concatenate((X_numerical, Y_numerical)), axis=0)
    numericals_ranges = numericals_maxs - numericals_mins

    X_rows = X_categorical.shape[0]

    """
    Perform a parallel calculation on a DataFrame by dividing the data into chunks
    and distributing the chunks across multiple CPUs (all the ones available CPUs except one).
    divide the total number of rows in X by the number of CPUs used to determine the size of the chunks on 
    which the parallel calculation is performed. 
    The for loop with index executes the actual calculation, and index is passed to fill the value that 
    corresponds to the distance from the same instance in case of calculation on a single DataFrame 
    (fill_diagonal == True).
    """
    if parallel:
        result_objs = []
        number_of_cpus = os.cpu_count() - 1
        chunk_size = int(X_rows / number_of_cpus)
        chunk_size = chunk_size if chunk_size > 0 else 1
        with Pool(processes=number_of_cpus) as pool:
            for index in range(0, X_rows, chunk_size):
                result = pool.apply_async(
                    _gower_matrix,
                    (
                        X_categorical[index : index + chunk_size],
                        X_numerical[index : index + chunk_size],
                        Y_categorical,
                        Y_numerical,
                        numericals_ranges,
                        weight_sum,
                        fill_diagonal,
                        index,
                    ),
                )
                result_objs.append(result)
            results = [result.get() for result in result_objs]
        dcr = np.concatenate(results)
    else:
        dcr = _gower_matrix(
                    X_categorical,
                    X_numerical,
                    Y_categorical,
                    Y_numerical,
                    numericals_ranges,
                    weight_sum,
                    fill_diagonal,
                )
    if save_output:
        _save_to_json("dcr_"+dcr_name, dcr)
    return dcr
    
def dcr_stats(dcr_name: str, distances_to_closest_record: np.ndarray) -> Dict:
    """
    This function returns the statisitcs for an array containing DCR computed previously.

    Parameters
    ----------
    dcr_name : str
        Name with which the DCR will be saved with in the JSON file used to generate the final report.
        Can be one of the following:
            - synth_train
            - synth_val
            - other
    distances_to_closest_record : np.ndarray
        A 1D-array containing the Distance to the Closest Record for each row of a dataframe
        shape (dataframe rows, )

    Returns
    -------
    Dict
        A dictionary containing mean and percentiles of the given DCR array.
    """
    if dcr_name != "synth_train" and dcr_name != "synth_val" and dcr_name != "other":
        raise TypeError("dcr_name must be one of the following:\n    -\"synth_train\"\n    -\"synth_val\"\n    -\"other\"")

    dcr_mean = np.mean(distances_to_closest_record)
    dcr_percentiles = np.percentile(distances_to_closest_record, [0, 25, 50, 75, 100])
    dcr_stats = {
        "mean": dcr_mean.item(),
        "min": dcr_percentiles[0].item(),
        "25%": dcr_percentiles[1].item(),
        "median": dcr_percentiles[2].item(),
        "75%": dcr_percentiles[3].item(),
        "max": dcr_percentiles[4].item(),
    }
    _save_to_json("dcr_"+dcr_name+"_stats", dcr_stats)
    return dcr_stats

def number_of_dcr_equal_to_zero(dcr_name: str, distances_to_closest_record: np.ndarray) -> int_type:
    """
    Return the number of 0s in the given DCR array, that is the number of duplicates/clones detected.

    Parameters
    ----------
    distances_to_closest_record : np.ndarray
        A 1D-array containing the Distance to the Closest Record for each row of a dataframe
        shape (dataframe rows, )

    Returns
    -------
    int
        The number of 0s in the given DCR array.
    """
    if dcr_name != "synth_train" and dcr_name != "synth_val" and dcr_name != "other":
        raise TypeError("dcr_name must be one of the following:\n    -\"synth_train\"\n    -\"synth_val\"\n    -\"other\"")

    zero_values_mask = distances_to_closest_record == 0.0
    _save_to_json("dcr_"+dcr_name+"_num_of_zeros", zero_values_mask.sum())
    return zero_values_mask.sum()

def dcr_histogram(
            dcr_name: str,
            distances_to_closest_record: np.ndarray, 
            bins: int = 20, 
            scale_to_100: bool = True
        ) -> Dict:
    """
    Compute the histogram of a DCR array: the DCR values equal to 0 are extracted before the
    histogram computation so that the first bar represent only the 0 (duplicates/clones)
    and the following bars represent the standard bins (with edge) of an histogram.

    Parameters
    ----------
    distances_to_closest_record : np.ndarray
        A 1D-array containing the Distance to the Closest Record for each row of a dataframe
        shape (dataframe rows, )
    bins : int, optional
        _description_, by default 20
    scale_to_100 : bool, optional
        Wheter to scale the histogram bins between 0 and 100 (instead of 0 and 1), by default True

    Returns
    -------
    Dict
        A dict containing the following items:
            * bins, histogram bins detected as string labels.
              The first bin/label is 0 (duplicates/clones), then the format is [inf_edge, sup_edge).
            * count, histogram values for each bin in bins
            * bins_edge_without_zero, the bin edges as returned by the np.histogram function without 0.
    """
    if dcr_name != "synth_train" and dcr_name != "synth_val" and dcr_name != "other":
        raise TypeError("dcr_name must be one of the following:\n    -\"synth_train\"\n    -\"synth_val\"\n    -\"other\"")

    range_bins_with_zero = ["0.0"]
    number_of_dcr_zeros = number_of_dcr_equal_to_zero(dcr_name, distances_to_closest_record)
    dcr_non_zeros = distances_to_closest_record[distances_to_closest_record > 0]
    counts_without_zero, bins_without_zero = np.histogram(
        dcr_non_zeros, bins=bins, range=(0.0, 1.0), density=False
    )
    if scale_to_100:
        scaled_bins_without_zero = bins_without_zero * 100
    else:
        scaled_bins_without_zero = bins_without_zero

    range_bins_with_zero.append("(0.0-{:.2f})".format(scaled_bins_without_zero[1]))
    for i, left_edge in enumerate(scaled_bins_without_zero[1:-2]):
        range_bins_with_zero.append(
            "[{:.2f}-{:.2f})".format(left_edge, scaled_bins_without_zero[i + 2])
        )
    range_bins_with_zero.append(
        "[{:.2f}-{:.2f}]".format(
            scaled_bins_without_zero[-2], scaled_bins_without_zero[-1]
        )
    )

    counts_with_zero = np.insert(counts_without_zero, 0, number_of_dcr_zeros)

    dcr_hist = {
        "bins": range_bins_with_zero,
        "counts": counts_with_zero.tolist(),
        "bins_edge_without_zero": bins_without_zero.tolist(),
    }
    
    _save_to_json("dcr_"+dcr_name+"_hist", dcr_hist)
    return dcr_hist

def validation_dcr_test(
                dcr_synth_train: np.ndarray, 
                dcr_synth_validation: np.ndarray
            ) -> float_type:
    """
    - If the returned percentage is close to (or smaller than) 50%, then the synthetic datset's records are equally close to the original training set and to the validation set.
      In this casse the synthetic data does not allow to conjecture whether a record was or was not contained in the training dataset.
    - If the returned percentage is greater than 50%, then the synthetic datset's records are closer to the training set than to the validation set, indicating 
      that vulnerable records are present in the synthetic dataset.

    Parameters
    ----------
    dcr_synth_train : np.ndarray
        A 1D-array containing the Distance to the Closest Record for each row of the synthetic
        dataset wrt the training dataset, shape (synthetic rows, )
    dcr_synth_validation : np.ndarray
        A 1D-array containing the Distance to the Closest Record for each row of the synthetic
        dataset wrt the validation dataset, shape (synthetic rows, )

    Returns
    -------
    float
        The percentage of synthetic rows closer to the training dataset than to the validation dataset.

    Raises
    ------
    ValueError
        If the two DCR array given as parameters have different shapes.
    """
    if dcr_synth_train.shape != dcr_synth_validation.shape:
        raise ValueError("Dcr arrays have different shapes.")

    warnings = ""
    percentage = 0.0

    if dcr_synth_train.sum() == 0:
        percentage = 100.0
        warnings = (
            "The synthetic dataset is an exact copy/clone of the training dataset."
        )
    elif (dcr_synth_train == dcr_synth_validation).all():
        percentage = 0.0
        warnings = (
            "The validation dataset is an exact copy/clone of the training dataset."
        )
    else:
        if dcr_synth_validation.sum() == 0:
            warnings = "The synthetic dataset is an exact copy/clone of the validation dataset."

        number_of_rows = dcr_synth_train.shape[0]
        synth_dcr_smaller_than_holdout_dcr_mask = dcr_synth_train < dcr_synth_validation
        synth_dcr_smaller_than_holdout_dcr_sum = (
            synth_dcr_smaller_than_holdout_dcr_mask.sum()
        )
        percentage = synth_dcr_smaller_than_holdout_dcr_sum / number_of_rows * 100
    
    dcr_validation = {"percentage": round(percentage,4), "warnings": warnings}
    _save_to_json("dcr_validation", dcr_validation)
    return dcr_validation