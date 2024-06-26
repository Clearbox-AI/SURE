import pyximport
import os
from multiprocessing import Pool

import numpy  as np
import pandas as pd
import polars as pl
import polars.selectors as cs

from typing import Dict, List, Tuple

pyximport.install(setup_args={"include_dirs": np.get_include()})
from sure.distance_metrics.gower_matrix_c import gower_matrix_c

def _polars_to_pandas(dataframe: pl.DataFrame | pl.LazyFrame):
    if isinstance(dataframe, pl.DataFrame):
        dataframe_pd = dataframe.to_pandas()
    if isinstance(dataframe, pl.LazyFrame):
        dataframe_pd = dataframe.collect().to_pandas()
    return dataframe_pd

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
                        x_dataframe: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
                        categorical_features: np.ndarray | List | Tuple,
                        y_dataframe: pd.DataFrame | pl.DataFrame | pl.LazyFrame = None,
                        feature_weights: np.ndarray | List = None,
                        parallel: bool = True,
                    ) -> np.ndarray:
    """
    Compute distance matrix between two dataframes containing mixed datatypes
    (numerical and categorical) using a modified version of the Gower's distance.

    Paper references:
    * A General Coefficient of Similarity and Some of Its Properties, J. C. Gower
    * Dimensionality Invariant Similarity Measure, Ahmad Basheer Hassanat

    Parameters
    ----------
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


    Returns
    -------
    np.ndarray
        1D array containing the Distance to the Closest Record for each row of x_dataframe
        shape (x_dataframe rows, )

    Raises
    ------
    TypeError
        If X and Y don't have the same (number of) columns.
    """
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

    if not isinstance(X, np.ndarray):
        if not np.array_equal(X.columns, Y.columns):
            raise TypeError("X and Y dataframes have different columns.")
    else:
        if not X.shape[1] == Y.shape[1]:
            raise TypeError("X and Y arrays have different number of columns.")

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
        return np.concatenate(results)
    else:
        return _gower_matrix(
            X_categorical,
            X_numerical,
            Y_categorical,
            Y_numerical,
            numericals_ranges,
            weight_sum,
            fill_diagonal,
        )