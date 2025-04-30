from typing import Dict, Tuple, Callable, List

import numpy  as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import random
from functools import reduce
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from clearbox_preprocessor import Preprocessor

from sure import _save_to_json
from sure._lazypredict import LazyClassifier, LazyRegressor

def _to_polars_df(df):
    """Converting Real and Synthetic Dataset into pl.DataFrame"""
    if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)
    elif isinstance(df, np.ndarray):
            df = pl.from_numpy(df)
    elif isinstance(df, pl.LazyFrame):
            df = df.collect()
    elif isinstance(df, pl.DataFrame):
        pass
    else:
        raise TypeError("Invalid type for dataframe")
    return df

def _to_numpy(data):
    ''' This functions transforms polars or pandas DataFrames or LazyFrames into numpy arrays'''
    if isinstance(data, pl.LazyFrame):
        return  data.collect().to_numpy()
    elif isinstance(data, pl.DataFrame | pd.DataFrame| pl.Series | pd.Series):
        return data.to_numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        print("The dataframe must be a Polars LazyFrame or a Pandas DataFrame")
        return
    
def _drop_cols(synth, real):
    ''' This function returns the real dataset without the columns that are not present in the synthetic one
    '''
    if not isinstance(synth, np.ndarray) and not isinstance(real, np.ndarray):
        col_synth    = set(synth.columns)
        col_real     = set(real.columns)
        not_in_real  = col_synth-col_real
        not_in_synth = col_real-col_synth

        # Drop columns that are present in the real dataset but are missing in the synthetic one
        if len(not_in_real)>0:
            print(f"""Warning: The following columns of the synthetic dataset are not present in the real dataset and were dropped to carry on with the computation:\n{not_in_real}\nIf you used only a subset of the dataset for computation, consider increasing the number of rows to ensure that all categorical values are adequately represented after one-hot-encoding.""")
        if isinstance(synth, pd.DataFrame):
            synth = synth.drop(columns=list(not_in_real))
        if isinstance(synth, pl.DataFrame):
            synth = synth.drop(list(not_in_real))
        if isinstance(synth, pl.LazyFrame):
            synth = synth.collect.drop(list(not_in_real))

        # Drop columns that are present in the real dataset but are missing in the synthetic one
        if isinstance(real, pd.DataFrame):
            real = real.drop(columns=list(not_in_synth))
        if isinstance(real, pl.DataFrame):
            real = real.drop(list(not_in_synth))
        if isinstance(real, pl.LazyFrame):
            real = real.collect.drop(list(not_in_synth))
    return synth, real
     
# MODEL GARDEN MODULE
class ClassificationGarden:
    """
    A class to facilitate the training and evaluation of multiple classification models 
    using LazyClassifier.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level, by default 0.
    ignore_warnings : bool, optional
        Whether to ignore warnings, by default True.
    custom_metric : callable, optional
        Custom metric function for evaluation, by default None.
    predictions : bool, optional
        Whether to return predictions along with model performance, by default False.
    classifiers : str or list, optional
        List of classifiers to use, or "all" for using all available classifiers, by default "all".

    :meta private:
    """

    def __init__(
        self,
        verbose         = 0,
        ignore_warnings = True,
        custom_metric   = None,
        predictions     = False,
        classifiers     = "all",
    ):
        self.verbose            = verbose
        self.ignore_warnings    = ignore_warnings
        self.custom_metric      = custom_metric
        self.predictions        = predictions
        self.classifiers        = classifiers

        self.clf = LazyClassifier(verbose         = verbose,
                                  ignore_warnings = ignore_warnings,
                                  custom_metric   = custom_metric,
                                  predictions     = predictions,
                                  classifiers     = classifiers)
        
    def fit(
        self, 
        X_train: pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
        X_test:  pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
        y_train: pl.DataFrame | pl.LazyFrame | pl.Series    | pd.Series | pd.DataFrame | np.ndarray, 
        y_test:  pl.DataFrame | pl.LazyFrame | pl.Series    | pd.Series | pd.DataFrame | np.ndarray
    ) ->     pd.DataFrame | np.ndarray:
        """
        Fit multiple classification models on the provided training data and evaluate on the test data.

        Parameters
        ----------
        X_train : pl.DataFrame, pl.LazyFrame, pd.DataFrame, or np.ndarray
            Training features.
        X_test : pl.DataFrame, pl.LazyFrame, pd.DataFrame, or np.ndarray
            Test features.
        y_train : pl.DataFrame, pl.LazyFrame, pl.Series, pd.Series, pd.DataFrame, or np.ndarray
            Training labels.
        y_test : pl.DataFrame, pl.LazyFrame, pl.Series, pd.Series, pd.DataFrame, or np.ndarray
            Test labels.

        Returns
        -------
        pd.DataFrame or np.ndarray
            Model performance metrics and predictions if specified.

        :meta private:
        """
        data = [X_train, X_test, y_train, y_test]
        
        for count, el in enumerate(data):
            data[count] = _to_numpy(el)

        models, predictions = self.clf.fit(data[0], data[1], data[2], data[3])
        return models, predictions

class RegressionGarden():
    """
    A class to facilitate the training and evaluation of multiple regression models 
    using LazyRegressor.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level, by default 0.
    ignore_warnings : bool, optional
        Whether to ignore warnings, by default True.
    custom_metric : callable, optional
        Custom metric function for evaluation, by default None.
    predictions : bool, optional
        Whether to return predictions along with model performance, by default False.
    regressors : str or list, optional
        List of regressors to use, or "all" for using all available regressors, by default "all".

    :meta private:
    """
    def __init__(
        self,
        verbose         = 0,
        ignore_warnings = True,
        custom_metric   = None,
        predictions     = False,
        regressors      = "all"
    ):
        self.verbose            = verbose
        self.ignore_warnings    =ignore_warnings
        self.custom_metric      = custom_metric
        self.predictions        = predictions
        self.regressors         = regressors

        self.reg = LazyRegressor(verbose            = verbose,
                                 ignore_warnings    = ignore_warnings,
                                 custom_metric      = custom_metric,
                                 predictions        = predictions,
                                 regressors         = regressors)

    def fit(
        self, 
        X_train: pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
        X_test:  pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
        y_train: pl.DataFrame | pl.LazyFrame | pl.Series    | pd.Series | pd.DataFrame | np.ndarray, 
        y_test:  pl.DataFrame | pl.LazyFrame | pl.Series    | pd.Series | pd.DataFrame | np.ndarray
    ) ->     pd.DataFrame | np.ndarray:
        """
        Fit multiple regression models on the provided training data and evaluate on the test data.

        Parameters
        ----------
        X_train : pl.DataFrame, pl.LazyFrame, pd.DataFrame, or np.ndarray
            Training features.
        X_test : pl.DataFrame, pl.LazyFrame, pd.DataFrame, or np.ndarray
            Test features.
        y_train : pl.DataFrame, pl.LazyFrame, pl.Series, pd.Series, pd.DataFrame, or np.ndarray
            Training labels.
        y_test : pl.DataFrame, pl.LazyFrame, pl.Series, pd.Series, pd.DataFrame, or np.ndarray
            Test labels.

        Returns
        -------
        pd.DataFrame or np.ndarray
            Model performance metrics and predictions if specified.

        :meta private:
        """
        data = [X_train, X_test, y_train, y_test]

        for count, el in enumerate(data):
            data[count] = _to_numpy(el)
            
        models, predictions = self.reg.fit(data[0], data[1], data[2], data[3])
        return models, predictions
    
# ML UTILITY METRICS MODULE
def compute_utility_metrics_class( 
    X_train:       pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
    X_synth:       pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray,                                   
    X_test:        pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
    y_train:       pl.DataFrame | pl.LazyFrame | pl.Series    | pd.Series | pd.DataFrame | np.ndarray,  
    y_synth:       pl.DataFrame | pl.LazyFrame | pl.Series    | pd.Series | pd.DataFrame | np.ndarray,  
    y_test:        pl.DataFrame | pl.LazyFrame | pl.Series    | pd.Series | pd.DataFrame | np.ndarray, 
    custom_metric: Callable       = None, 
    classifiers:   List[Callable] = "all",
    predictions:   bool = False,
    path_to_json:  str = ""
): 
    """
    Train and evaluate classification models on both real and synthetic datasets.

    Parameters
    ----------
    X_train : Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, np.ndarray]
        Training features for real data.
    X_synth : Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, np.ndarray]
        Training features for synthetic data.
    X_test : Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, np.ndarray]
        Test features for evaluation.
    y_train : Union[pl.DataFrame, pl.LazyFrame, pl.Series, pd.Series, pd.DataFrame, np.ndarray]
        Training labels for real data.
    y_synth : Union[pl.DataFrame, pl.LazyFrame, pl.Series, pd.Series, pd.DataFrame, np.ndarray]
        Training labels for synthetic data.
    y_test : Union[pl.DataFrame, pl.LazyFrame, pl.Series, pd.Series, pd.DataFrame, np.ndarray]
        Test labels for evaluation.
    custom_metric : Callable, optional
        Custom metric for model evaluation, by default None.
    classifiers : List[Callable], optional
        List of classifiers to use, or "all" for all available classifiers, by default "all".
    predictions : bool, optional
        If True, returns predictions along with model performance, by default False.
    path_to_json : str, optional
        Path to save the output JSON files, by default "".

    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame, np.ndarray]
        Model performance metrics for both real and synthetic datasets, and optionally predictions.
    """
    # Store DataFrame type information for returning the same type
    was_pd = True
    was_pl = False
    was_np = False

    if isinstance(X_train, pl.DataFrame) or isinstance(X_train, pl.LazyFrame):
         was_pl = True
    elif isinstance(X_train, np.ndarray):
         was_np = True

    # Initialise ClassificationGarden class and start training
    classifier = ClassificationGarden(predictions=predictions, classifiers=classifiers, custom_metric=custom_metric)
    print('Fitting original models:')
    models_train, pred_train = classifier.fit(X_train[[w for w in X_train.columns if w in X_test.columns]], 
                                  X_test[[w for w in X_train.columns if w in X_test.columns]], 
                                  y_train, 
                                  y_test)
                                     
    print('Fitting synthetic models:')
    classifier_synth = ClassificationGarden(predictions=predictions, classifiers=classifiers, custom_metric=custom_metric)
    models_synth, pred_synth = classifier_synth.fit(X_synth[[w for w in X_synth.columns if w in X_test.columns]], 
                                                    X_test[[w for w in X_synth.columns if w in X_test.columns]], 
                                                    y_synth, 
                                                    y_test)
    delta = models_train-models_synth
    col_names_delta = [s + " Delta" for s in list(delta.columns)]
    delta.columns = col_names_delta
    col_names_train= [s + " Real" for s in list(models_train.columns)]
    models_train.columns = col_names_train
    col_names_synth = [s + " Synth" for s in list(models_synth.columns)]
    models_synth.columns = col_names_synth
    
    _save_to_json("models", models_train, path_to_json)
    _save_to_json("models_synth", models_synth, path_to_json)
    _save_to_json("models_delta", delta, path_to_json)
    
    # Transform the output DataFrames into the type used for the input DataFrames
    if was_pl:
         models_train = pl.from_pandas(models_train)
         models_synth = pl.from_pandas(models_synth)
         delta = pl.from_pandas(delta)
    elif was_np:
         models_train = pl.from_numpy(models_train)
         models_synth = pl.from_numpy(models_synth)
         delta = pl.from_numpy(delta)

    if predictions:
        if isinstance(y_train, (pl.DataFrame, pl.LazyFrame, pl.Series)):
            y_train = y_train.to_pandas()
            y_synth = y_synth.to_pandas()
        elif isinstance(y_train, np.ndarray):
            y_train = pd.DataFrame(y_train)
            y_synth = pd.DataFrame(y_synth)

        pred_train = pd.concat([y_train,pred_train], axis=1)
        pred_train.columns.values[0] = 'Ground truth'
        pred_synth = pd.concat([y_synth,pred_synth], axis=1)
        pred_synth.columns.values[0] = 'Ground truth'

        if was_pl:
            pred_train = pl.from_pandas(pred_train)
            pred_synth = pl.from_pandas(pred_synth)
        elif was_np:
            pred_train = pred_train.to_numpy()
            pred_synth = pred_synth.to_numpy()
        return models_train, models_synth, delta, pred_train, pred_synth
    else:
        return models_train, models_synth, delta

def compute_utility_metrics_regr(
    X_train:       pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
    X_synth:       pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray,                                   
    X_test:        pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
    y_train:       pl.DataFrame | pl.LazyFrame | pl.Series    | pd.Series | pd.DataFrame | np.ndarray,  
    y_synth:       pl.DataFrame | pl.LazyFrame | pl.Series    | pd.Series | pd.DataFrame | np.ndarray,  
    y_test:        pl.DataFrame | pl.LazyFrame | pl.Series    | pd.Series | pd.DataFrame | np.ndarray, 
    custom_metric: Callable       = None, 
    regressors:    List[Callable] = "all",
    predictions:   bool = False,
    path_to_json:  str = ""
):
    """
    Train and evaluate regression models on both real and synthetic datasets.

    Parameters
    ----------
    X_train : Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, np.ndarray]
        Training features for real data.
    X_synth : Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, np.ndarray]
        Training features for synthetic data.
    X_test : Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, np.ndarray]
        Test features for evaluation.
    y_train : Union[pl.DataFrame, pl.LazyFrame, pl.Series, pd.Series, pd.DataFrame, np.ndarray]
        Training labels for real data.
    y_synth : Union[pl.DataFrame, pl.LazyFrame, pl.Series, pd.Series, pd.DataFrame, np.ndarray]
        Training labels for synthetic data.
    y_test : Union[pl.DataFrame, pl.LazyFrame, pl.Series, pd.Series, pd.DataFrame, np.ndarray]
        Test labels for evaluation.
    custom_metric : Callable, optional
        Custom metric for model evaluation, by default None.
    regressors : List[Callable], optional
        List of regressors to use, or "all" for all available regressors, by default "all".
    predictions : bool, optional
        If True, returns predictions along with model performance, by default False.
    path_to_json : str, optional
        Path to save the output JSON files, by default "".

    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame, np.ndarray]
        Model performance metrics for both real and synthetic datasets, and optionally predictions.
    """
    # Store DataFrame type information for returning the same type
    was_pd = True
    was_pl = False
    was_np = False

    if isinstance(X_train, pl.DataFrame) or isinstance(X_train, pl.LazyFrame):
         was_pl = True
    elif isinstance(X_train, np.ndarray):
         was_np = True

    # Initialise RegressionGarden class and start training
    regressor = RegressionGarden(predictions=predictions, regressors=regressors, custom_metric=custom_metric)
    print('Fitting original models:')
    models_train, pred_train = regressor.fit(X_train[[w for w in X_train.columns if w in X_test.columns]], 
                                 X_test[[w for w in X_train.columns if w in X_test.columns]], 
                                 y_train, 
                                 y_test)
    print('Fitting synthetic models:')

    regressor_synth = RegressionGarden(predictions=predictions, regressors=regressors, custom_metric=custom_metric)
    models_synth, pred_synth = regressor_synth.fit(X_synth[[w for w in X_synth.columns if w in X_test.columns]], 
                                                   X_test[[w for w in X_synth.columns if w in X_test.columns]], 
                                                   y_synth, 
                                                   y_test)
    delta = models_train-models_synth
    col_names_delta = [s + " Delta" for s in list(delta.columns)]
    delta.columns = col_names_delta
    col_names_train= [s + " Real" for s in list(models_train.columns)]
    models_train.columns = col_names_train
    col_names_synth = [s + " Synth" for s in list(models_synth.columns)]
    models_synth.columns = col_names_synth

    _save_to_json("models", models_train, path_to_json)
    _save_to_json("models_synth", models_synth, path_to_json)
    _save_to_json("models_delta", delta, path_to_json)

    pred_train = pd.concat([y_train.to_pandas(),pred_train], axis=1)
    pred_train.columns.values[0] = 'Ground truth'
    pred_synth = pd.concat([y_synth.to_pandas(),pred_synth], axis=1)
    pred_synth.columns.values[0] = 'Ground truth'

    # Transform the output DataFrames into the type used for the input DataFrames
    if isinstance(y_train, (pl.DataFrame, pl.LazyFrame)):
        y_train = y_train.to_pandas()
        y_synth = y_synth.to_pandas()
    elif isinstance(y_train, np.ndarray):
        y_train = pd.DataFrame(y_train)
        y_synth = pd.DataFrame(y_synth)

    pred_train = pd.concat([y_train,pred_train], axis=1)
    pred_train.columns.values[0] = 'Groud truth'
    pred_synth = pd.concat([y_synth,pred_synth], axis=1)
    pred_synth.columns.values[0] = 'Groud truth'    
    
    if was_pl:
         models_train = pl.from_pandas(models_train)
         models_synth = pl.from_pandas(models_synth)
         delta = pl.from_pandas(delta)
    elif was_np:
         models_train = pl.from_numpy(models_train)
         models_synth = pl.from_numpy(models_synth)
         delta = pl.from_numpy(delta)

    if predictions:
        if was_pl:
            pred_train = pl.from_pandas(pred_train)
            pred_synth = pl.from_pandas(pred_synth)
        elif was_np:
            pred_train = pl.from_numpy(pred_train)
            pred_synth = pl.from_numpy(pred_synth)
        return models_train, models_synth, delta, pred_train, pred_synth
    else:
        return models_train, models_synth, delta

# STATISTICAL SIMILARITIES METRICS MODULE
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
        most_frequent = data[feature].mode().cast(pl.String).to_list()

        # Create the dictionary entry for the current feature
        if len(most_frequent)>5:
            most_frequent_dict[feature] = None
        else:
            most_frequent_dict[feature] = most_frequent
    
    return most_frequent_dict

def compute_statistical_metrics(
    real_data:    pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
    synth_data:   pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray,
    path_to_json: str = ""
) -> Tuple[Dict, Dict, Dict]:
    """
    Compute statistical metrics for numerical, categorical, and temporal features
    in both real and synthetic datasets.

    Parameters
    ----------
    real_data : Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, np.ndarray]
        The real dataset containing numerical, categorical, and/or temporal features.
    synth_data : Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, np.ndarray]
        The synthetic dataset containing numerical, categorical, and/or temporal features.
    path_to_json : str, optional
        The file path to save the comparison metrics in JSON format, by default "".

    Returns
    -------
    Tuple[Dict, Dict, Dict]
        A tuple containing three dictionaries with statistical comparisons for
        numerical, categorical, and temporal features, respectively.
    """
    num_features_comparison  = None
    cat_features_comparison  = None
    time_features_comparison = None

    # Converting Real and Synthetic Dataset into pl.DataFrame
    real_data = _to_polars_df(real_data)
    synth_data = _to_polars_df(synth_data)

    # Drop columns that are present in the real dataset but not in the synthetic dataset and vice versa
    synth_data, real_data = _drop_cols(synth_data, real_data)

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
        num_features_comparison["skewness"]         = { "real"      : real_data.select(pl.col(num_features).skew()),
                                                        "synthetic" : synth_data.select(pl.col(num_features).skew())}
        num_features_comparison["kurtosis"]         = { "real"      : real_data.select(pl.col(num_features).kurtosis()),
                                                        "synthetic" : synth_data.select(pl.col(num_features).kurtosis())}

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

    _save_to_json("num_features_comparison", num_features_comparison, path_to_json)
    _save_to_json("cat_features_comparison", cat_features_comparison, path_to_json)
    _save_to_json("time_features_comparison", time_features_comparison, path_to_json)

    return num_features_comparison, cat_features_comparison, time_features_comparison

def compute_mutual_info(
    real_data:  pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
    synth_data: pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray,
    exclude_columns: List = [],
    path_to_json: str = ""
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Compute the correlation matrices for both real and synthetic datasets, and
    calculate the difference between these matrices.

    Parameters
    ----------
    real_data : Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, np.ndarray]
        The real dataset, which can be in the form of a Polars DataFrame, LazyFrame,
        pandas DataFrame, or numpy ndarray.
    synth_data : Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, np.ndarray]
        The synthetic dataset, provided in the same format as `real_data`.
    exclude_columns: List, option
        A list of columns to exclude from the computaion of mutual information,
        by default [].
    path_to_json : str, optional
        File path to save the correlation matrices and their differences in JSON format,
        by default "".

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        A tuple containing:
        - real_corr: Correlation matrix of the real dataset with column names included.
        - synth_corr: Correlation matrix of the synthetic dataset with column names included.
        - diff_corr: Difference between the correlation matrices of the real and synthetic
          datasets, with values smaller than 1e-5 substituted with 0.

    Raises
    ------
    ValueError
        If the features in the real and synthetic datasets do not match or if non-numerical
        features are present.
    """
    # Converting Real and Synthetic Dataset into pl.DataFrame
    real_data = _to_polars_df(real_data)
    synth_data = _to_polars_df(synth_data)
    
    for col in exclude_columns:
        if col not in real_data.columns:
            raise KeyError(f"Column {col} not found in DataFrame.")

    real_data  = real_data.drop(exclude_columns)
    synth_data = synth_data.drop(exclude_columns)

    # Drop columns that are present in the real dataset but not in the synthetic dataset and vice versa
    synth_data, real_data = _drop_cols(synth_data, real_data)

    # Check that the real features and the synthetic ones match
    if not real_data.columns==synth_data.columns:
        raise ValueError("The features from the real dataset and the synthetic one do not match.")
    
    # Convert Boolean and Temporal types to numerical
    real_data   = real_data.with_columns(cs.boolean().cast(pl.UInt8))
    synth_data  = synth_data.with_columns(cs.boolean().cast(pl.UInt8))
    real_data   = real_data.with_columns(cs.temporal().as_expr().dt.timestamp('ms'))
    synth_data  = synth_data.with_columns(cs.temporal().as_expr().dt.timestamp('ms'))
    
    # Label Encoding of categorical features to compute mutual information
    encoder = LabelEncoder()
    for col in real_data.select(cs.string()).columns:
        real_data = real_data.with_columns([
            pl.Series(col, encoder.fit_transform(real_data[col].to_list())),
        ])
        synth_data = synth_data.with_columns([
            pl.Series(col, encoder.fit_transform(synth_data[col].to_list())),
        ])

    # Real and Synthetic dataset correlation matrix
    real_corr = real_data.corr()
    synth_corr = synth_data.corr()

    # Difference between the correlation matrix of the real dataset and the correlation matrix of the synthetic dataset
    diff_corr = real_corr-synth_corr

    # Substitute elements with abs value lower than 1e-5 with 0
    diff_corr = diff_corr.with_columns([pl.when(abs(pl.col(col)) < 1e-5).then(0).otherwise(pl.col(col)).alias(col) for col in diff_corr.columns])

    _save_to_json("real_corr", real_corr, path_to_json)
    _save_to_json("synth_corr", synth_corr, path_to_json)
    _save_to_json("diff_corr", diff_corr, path_to_json)

    return real_corr, synth_corr, diff_corr

def detection_score(
        df_original:  pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
        df_synthetic:  pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray,  
        preprocessor: Preprocessor = None, 
        features_to_hide: List = [], 
        path_to_json: str = ""
    ) -> Dict:
    """
    Computes the detection score by training an XGBoost model to differentiate between
    original and synthetic data. The lower the model's accuracy, the higher the quality
    of the synthetic data.

    Parameters
    ----------
    features_to_hide : list, optional
        List of features to exclude from importance analysis. Defaults to an empty list.

    Returns
    -------
    dict
        A dictionary containing accuracy, ROC AUC score, detection score, and feature importances.

    Notes
    -----
    The method operates through the following steps:

    1. Prepares the datasets:

        - Samples the original dataset to match the size of the synthetic dataset.
        - Preprocesses both datasets to ensure consistent feature representation.
        - Labels original data as `0` and synthetic data as `1`.
    
    2. Builds a classification model:

        - Uses XGBoost to train a model that classifies data points as either real or synthetic.
        - Splits the data into training and test sets.
        - Trains the model using a 33% test split.

    3. Computes Evaluation Metrics:

        - Accuracy: Measures classification correctness.
        - ROC-AUC Score: Measures the model’s discriminatory power.
        - Detection Score
        
    4. Extracts Feature Importances:
    
        - Identifies which features contribute most to distinguishing real vs. synthetic data.
        - Helps detect which synthetic features deviate from real-world patterns.

    
    The detection score is calculated as:

    .. code-block:: python

        detection_score["score"] = (1 - detection_score["ROC_AUC"]) * 2

    - If ``ROC_AUC <= 0.5``, the synthetic data is considered indistinguishable (``score = 1``).
    - A lower score means better synthetic data quality.
    - Feature importance analysis helps detect which synthetic features deviate most from real data.

    Examples
    --------
    Example of dictionary returned:

    .. code-block:: python

        >>> detection_results = detection_score(original_df, synthetic_df)
        >>> print(detection_results)
        {
            "accuracy": 0.85,   # How often the model classifies correctly
            "ROC_AUC": 0.90,   # The ability to distinguish real vs synthetic
            "score": 0.2,     # The final detection score (lower = better)
            "feature_importances": {"feature_1": 0.34, "feature_2": 0.21, ...} 
        }

    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    if preprocessor is None:
        preprocessor = Preprocessor(df_original)

    df_original  = _to_polars_df(df_original)
    df_synthetic = _to_polars_df(df_synthetic)

    # Sample from the original dataset to match the size of the synthetic dataset
    df_original = df_original.head(len(df_synthetic))
    
    # Replace minority labels in the original data
    for i in preprocessor.discarded[1].keys():
        list_minority_labels = preprocessor.discarded[1][i]
        for j in list_minority_labels:
            df_original = df_original.with_columns(
                pl.when(pl.col(i) == j)
                .then(pl.lit("other"))
                .otherwise(pl.col(i))
                .alias(i)
            )

    # Preprocess and label the original data
    preprocessed_df_original = preprocessor.transform(df_original)
    df_original              = preprocessor.inverse_transform(preprocessed_df_original)
    df_original              = df_original.with_columns([
        pl.Series("label", np.zeros(len(df_original)).astype(int)),
    ])

    # Preprocess and label the synthetic data
    preprocessed_df_synthetic = preprocessor.transform(df_synthetic)
    df_synthetic = preprocessor.inverse_transform(preprocessed_df_synthetic)
    df_synthetic = df_synthetic.with_columns([
        pl.Series("label", np.ones(len(df_synthetic)).astype(int)),
    ])

    df = pl.concat([df_original, df_synthetic])
    preprocessor_ = Preprocessor(df, target_column = "label")
    df_preprocessed = preprocessor_.transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df_preprocessed.select(pl.exclude("label")), 
        df_preprocessed.select(pl.col("label")), 
        test_size=0.33, 
        random_state=42
    )

    model = xgb.XGBClassifier(max_depth=3, n_estimators=50, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Make predictions and compute metrics
    y_pred = model.predict(X_test)
    detection_score = {}
    detection_score["accuracy"] = round(
        accuracy_score(y_true=y_test, y_pred=y_pred), 4
    )
    detection_score["ROC_AUC"] = round(
        roc_auc_score(
            y_true=y_test,
            y_score=model.predict_proba(X_test)[:, 1],
            average=None,
        ),
        4,
    )
    detection_score["score"] = (
        1
        if detection_score["ROC_AUC"] <= 0.5
        else (1 - detection_score["ROC_AUC"]) * 2
    )

    detection_score["feature_importances"] = {}

    # Determine feature importances
    numerical_features_sizes, categorical_features_sizes = preprocessor.get_features_sizes()

    numerical_features = preprocessor.numerical_features
    categorical_features = preprocessor.categorical_features
    datetime_features = preprocessor.datetime_features

    index = 0
    for feature, importance in zip(numerical_features, model.feature_importances_):
        if feature not in features_to_hide:
            detection_score["feature_importances"][feature] = round(float(importance), 4)
        index += 1

    if len(datetime_features)>0:
        for feature, importance in zip(datetime_features, model.feature_importances_[index:]):
            if feature not in features_to_hide:
                detection_score["feature_importances"][feature] = round(float(importance), 4)
            index += 1

    for feature, feature_size in zip(categorical_features, categorical_features_sizes):
        importance = np.sum(model.feature_importances_[index : index + feature_size])
        index += feature_size
        if feature not in features_to_hide:
            detection_score["feature_importances"][feature] = round(float(importance), 4)
            
    _save_to_json("detection_score", detection_score, path_to_json)

    return detection_score

def query_power(
        df_original: pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray,
        df_synthetic: pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray,
        preprocessor: Preprocessor = None,
        path_to_json: str = ""
        ) -> dict:
    """
    Generates and runs queries to compare the original and synthetic datasets.

    This method creates random queries that filter data from both datasets.
    The similarity between the sizes of the filtered results is used to score
    the quality of the synthetic data.

    Returns:
        dict: A dictionary containing query texts, the number of matches for each
                query in both datasets, and an overall score indicating the quality
                of the synthetic data.
    """
    def polars_query(feat_type, feature, op, value):
        if feat_type == 'num':
            if op == "<=":
                query = pl.col(feature) <= value
            elif op == ">=":
                query = pl.col(feature) >= value
        elif feat_type == 'cat':
            if op == "==":
                query = pl.col(feature) == value
            elif op == "!=":
                query = pl.col(feature) != value
        return query
    
    query_power = {"queries": []}

    if preprocessor is None:
        preprocessor = Preprocessor(df_original)

    df_original  = _to_polars_df(df_original)
    df_synthetic = _to_polars_df(df_synthetic)

    df_original = df_original.sample(len(df_synthetic)).clone()
    df_original_preprocessed = preprocessor.transform(df_original)
    df_original = preprocessor.inverse_transform(df_original_preprocessed)

    df_synthetic_preprocessed = preprocessor.transform(df_synthetic)
    df_synthetic = preprocessor.inverse_transform(df_synthetic_preprocessed)

    # Extract feature types
    numerical_features = preprocessor.numerical_features
    categorical_features = preprocessor.categorical_features
    datetime_features = preprocessor.datetime_features
    boolean_features = preprocessor.boolean_features

    # Prepare the feature list, excluding datetime features
    features = list(set(df_original.columns) - set(datetime_features))

    # Define query parameters
    quantiles = [0.25, 0.5, 0.75]
    numerical_ops = ["<=", ">="]
    categorical_ops = ["==", "!="]
    logical_ops = ["and"]

    queries_score = []

    # Generate and run up to 5 queries
    while len(features) >= 2 and len(query_power["queries"]) < 5:
        # Randomly select two features for the query
        feats = [random.choice(features)]
        features.remove(feats[0])
        feats.append(random.choice(features))
        features.remove(feats[1])

        queries = []
        queries_text = []
        # Construct query conditions for each selected feature
        for feature in feats:
            if feature in numerical_features:
                feat_type = 'num'
                op = random.choice(numerical_ops)
                value = df_original.select(
                    pl.col(feature).quantile(
                        random.choice(quantiles), interpolation="nearest"
                    )).item()
            elif feature in categorical_features or feature in boolean_features:
                feat_type = 'cat'
                op = random.choice(categorical_ops)
                value = random.choice(df_original[feature].unique())
            else:
                continue

            queries_text.append(f"`{feature}` {op} `{value}`")
            queries.append(polars_query(feat_type, feature, op, value))

        # Combine query conditions with a logical operator
        text = f" {random.choice(logical_ops)} ".join(queries_text)
        combined_query = reduce(lambda a, b: a & b, queries)

        try:
            query = {
                "text": text,
                "original_df": len(df_original.filter(combined_query)),
                "synthetic_df": len(df_synthetic.filter(combined_query)),
            }
        except Exception:
            query = {"text": "Invalid query", "original_df": 0, "synthetic_df": 0}

        # Append the query and calculate the score
        query_power["queries"].append(query)
        queries_score.append(
            1 - abs(query["original_df"] - query["synthetic_df"]) / len(df_original)
        )

    # Calculate the overall query power score
    query_power["score"] = round(float(sum(queries_score) / len(queries_score)), 4)

    _save_to_json("query_power", query_power, path_to_json)

    return query_power