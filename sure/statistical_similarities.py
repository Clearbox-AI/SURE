import numpy  as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from typing import Dict, Tuple, List

def _value_count(data: pl.DataFrame, 
                 features: List
                ) -> Dict:
        ''' This function returns the unique values count and frequency for each feature in a Polars DataFrame
        '''
        values = dict()
        for feature in data.select(pl.col(features)).columns:
            # Get the value counts for the specified feature
            value_counts_df = data.group_by(feature).len(name="count")

            # Calculate the frequency
            total_counts = value_counts_df['count'].sum()
            value_counts_df = value_counts_df.with_columns(
                (pl.col('count') / total_counts * 100).round(2).alias('freq_%')
            ).sort("freq_%",descending=True)

            values[feature] = value_counts_df

        return values

def _most_frequent_values(data: pl.DataFrame,
                          features: List
                         ) -> Dict:
    ''' This function returns the most frequent value for each feature of the Polars DataFrame
    '''
    most_frequent_dict = dict()
    for feature in features:
        # Calculate the most frequent value (mode) for original dataset
        most_frequent = data[feature].mode().to_list()

        # Create the dictionary entry for the current feature
        if len(most_frequent)>5:
            most_frequent_dict[feature] = None
        else:
            most_frequent_dict[feature] = most_frequent
    
    return most_frequent_dict

def compute_statistical_metrics(real_data:  pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
                                synth_data: pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray) -> Tuple[Dict, Dict, Dict]:
    ''' This function computes statistical metrics for numerical, categorical, and temporal features 
        in both real and synthetic datasets.
        
        Parameters:
        - real_data: The real dataset, can be in the form of Polars DataFrame, LazyFrame, pandas DataFrame, or numpy ndarray.
        - synth_data: The synthetic dataset, in the same format as real_data.
        
        Returns:
        - num_features_comparison: Dictionary containing statistical metrics for numerical features.
        - cat_features_comparison: Dictionary containing statistical metrics for categorical features.
        - time_features_comparison: Dictionary containing statistical metrics for temporal features.
    '''
    num_features_comparison  = None
    cat_features_comparison  = None
    time_features_comparison = None

    # Converting Real and Synthetic Dataset in to pl.DataFrame
    if isinstance(real_data, pd.DataFrame):
            real_data = pl.from_pandas(real_data)
    if isinstance(synth_data, pd.DataFrame):
            synth_data = pl.from_pandas(synth_data)
    if isinstance(real_data, np.ndarray):
            real_data = pl.from_pandas(real_data)
    if isinstance(synth_data, np.ndarray):
            synth_data = pl.from_pandas(synth_data)
    if isinstance(real_data, pl.LazyFrame):
            real_data = real_data.collect()
    if isinstance(synth_data, pl.LazyFrame):
            synth_data = synth_data.collect()

    # Check that the real features and the synthetic ones match
    if not real_data.columns==synth_data.columns:
        raise ValueError("The features from the real dataset and the synthetic one do not match.")
    
    # Transform boolean into int
    real_data   = real_data.with_columns(pl.col(pl.Boolean).cast(pl.UInt8))
    synth_data  = synth_data.with_columns(pl.col(pl.Boolean).cast(pl.UInt8))
    
    cat_features        = cs.expand_selector(real_data, cs.string())
    num_features        = cs.expand_selector(real_data, cs.numeric())
    time_features       = cs.expand_selector(real_data, cs.temporal())

    # Numerical features
    if len(num_features) != 0:
        num_features_comparison = dict()
        num_features_comparison["null_count"]       = { "real"      : real_data.select(pl.col(num_features)).null_count(),
                                                        "synthetic" : synth_data.select(pl.col(num_features)).null_count()}
        num_features_comparison["unique_val_number"]= { "real"      : real_data.select(pl.n_unique(num_features)),
                                                        "synthetic" : synth_data.select(pl.n_unique(num_features))}
        num_features_comparison["mean"]             = { "real"      : real_data.select(pl.mean(num_features)),
                                                        "synthetic" : synth_data.select(pl.mean(num_features))}
        num_features_comparison["std"]              = { "real"      : real_data.select(num_features).std(),
                                                        "synthetic" : synth_data.select(num_features).std()}
        num_features_comparison["min"]              = { "real"      : real_data.select(pl.min(num_features)),
                                                        "synthetic" : synth_data.select(pl.min(num_features))}
        num_features_comparison["first_quartile"]   = { "real"      : real_data.select(num_features).quantile(0.25,"nearest"),
                                                        "synthetic" : synth_data.select(num_features).quantile(0.25,"nearest")}
        num_features_comparison["second_quartile"]  = { "real"      : real_data.select(num_features).quantile(0.5,"nearest"),
                                                        "synthetic" : synth_data.select(num_features).quantile(0.5,"nearest")}
        num_features_comparison["third_quartile"]   = { "real"      : real_data.select(num_features).quantile(0.75,"nearest"),
                                                        "synthetic" : synth_data.select(num_features).quantile(0.75,"nearest")}
        num_features_comparison["max"]              = { "real"      : real_data.select(pl.max(num_features)),
                                                        "synthetic" : synth_data.select(pl.max(num_features))}
    # Categorical features
    if len(cat_features) != 0:
        cat_features_comparison = dict()
        cat_features_comparison["null_count"]       = { "real"      : real_data.select(pl.col(cat_features)).null_count(),
                                                        "synthetic" : synth_data.select(pl.col(cat_features)).null_count()}
        cat_features_comparison["unique_val_number"]= { "real"      : real_data.select(pl.n_unique(cat_features)),
                                                        "synthetic" : synth_data.select(pl.n_unique(cat_features))}
        cat_features_comparison["unique_val_stats"] = { "real"      : _value_count(real_data, cat_features),
                                                        "synthetic" : _value_count(synth_data, cat_features)}
    
    # Temporal features
    if len(time_features) != 0:
        time_features_comparison = dict()
        time_features_comparison["null_count"]          = { "real"      : real_data.select(pl.col(time_features)).null_count(),
                                                            "synthetic" : synth_data.select(pl.col(time_features)).null_count()}
        time_features_comparison["unique_val_number"]   = { "real"      : real_data.select(pl.n_unique(time_features)),
                                                            "synthetic" : synth_data.select(pl.n_unique(time_features))}
        time_features_comparison["min"]                 = { "real"      : real_data.select(pl.min(time_features)),
                                                            "synthetic" : synth_data.select(pl.min(time_features))}
        time_features_comparison["max"]                 = { "real"      : real_data.select(pl.max(time_features)),
                                                            "synthetic" : synth_data.select(pl.max(time_features))}
        time_features_comparison["most_frequent"]       = { "real"      : _most_frequent_values(real_data, time_features),
                                                            "synthetic" : _most_frequent_values(synth_data, time_features)}
        
    return num_features_comparison, cat_features_comparison, time_features_comparison

def compute_mutual_info(real_data:  pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
                        synth_data: pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray):
    ''' This function computes the correlation matrix for both the real and synthetic datasets,
        and calculates the difference between these matrices.
        
        Parameters:
        - real_data: The real dataset, can be in the form of Polars DataFrame, LazyFrame, pandas DataFrame, or numpy ndarray.
        - synth_data: The synthetic dataset, in the same format as real_data.
        
        Returns:
        - real_corr: Correlation matrix of the real dataset with column names included.
        - synth_corr: Correlation matrix of the synthetic dataset with column names included.
        - diff_corr: Difference between the correlation matrices of real and synthetic datasets,
                        with values smaller than 1e-5 substituted with 0.
    '''
    # Converting Real and Synthetic Dataset into pl.DataFrame
    if isinstance(real_data, pd.DataFrame):
            real_data = pl.from_pandas(real_data)
    if isinstance(synth_data, pd.DataFrame):
            synth_data = pl.from_pandas(synth_data)
    if isinstance(real_data, np.ndarray):
            real_data = pl.from_numpy(real_data)
    if isinstance(synth_data, np.ndarray):
            synth_data = pl.from_numpy(synth_data)
    if isinstance(real_data, pl.LazyFrame):
            real_data = real_data.collect()
    if isinstance(synth_data, pl.LazyFrame):
            synth_data = synth_data.collect()

    # Check that the real features and the synthetic ones match
    if not real_data.columns==synth_data.columns:
        raise ValueError("The features from the real dataset and the synthetic one do not match.")
    
    # Convert Boolean and Temporal types to numerical
    real_data   = real_data.with_columns(cs.boolean().cast(pl.UInt8))
    synth_data  = synth_data.with_columns(cs.boolean().cast(pl.UInt8))
    real_data   = real_data.with_columns(cs.temporal().as_expr().dt.timestamp('ms'))
    synth_data  = synth_data.with_columns(cs.temporal().as_expr().dt.timestamp('ms'))
    
    # Check that only numerical features are present
    for col in real_data.columns:
          if not real_data[col].dtype.is_numeric():
            raise ValueError("Some non-numierical features is presnt in the dataset. Make sure to run the Preprocessor before evluating mutual information.")       
    
    # Real dataset correlation matrix
    real_corr = real_data.corr()

    # Synthetic dataset correlation matrix
    synth_corr = synth_data.corr()

    # Difference between the correlation matrix of the real dataset and the correlation matrix of the synthetic dataset
    diff_corr = real_corr-synth_corr

    # Substitute elements with abs value lower than 1e-5 with 0
    diff_corr = diff_corr.with_columns([pl.when(abs(pl.col(col)) < 1e-5).then(0).otherwise(pl.col(col)).alias(col) for col in diff_corr.columns])

    # Add the Column names column as first coulmn
    real_corr  = real_corr.insert_column(0, pl.Series("Columns", real_data.columns))
    synth_corr = synth_corr.insert_column(0, pl.Series("Columns", synth_corr.columns))
    diff_corr  = diff_corr.insert_column(0, pl.Series("Columns", diff_corr.columns))

    return real_corr, synth_corr, diff_corr