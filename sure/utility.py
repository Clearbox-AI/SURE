from typing import Dict, Tuple, Callable, List

import numpy  as np
import pandas as pd
import polars as pl
import polars.selectors as cs

from sure import _save_to_json
from sure._lazypredict import LazyClassifier, LazyRegressor

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
    
def _drop_real_cols(synth, real):
    ''' This function returns the real dataset without the columns that are not present in the synthetic one
    '''
    if not isinstance(synth, np.ndarray) and not isinstance(real, np.ndarray):
        col_synth    = set(synth.columns)
        col_real     = set(real.columns)
        not_in_real  = col_synth-col_real
        not_in_synth = col_real-col_synth

        if len(not_in_real)>0:
            print(f"The following columns of the synthetic dataset are not present in the real dataset:\n{not_in_real}")
            return

        # Drop columns that are present in the real dataset but are missing in the synthetic one
        if isinstance(real, pd.DataFrame):
            real = real.drop(columns=list(not_in_synth))
        if isinstance(real, pl.DataFrame):
            real = real.drop(list(not_in_synth))
        if isinstance(real, pl.LazyFrame):
            real = real.collect.drop(list(not_in_synth))
    return real
     
# MODEL GARDEN MODULE
class ClassificationGarden:
    def __init__(self,
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
        
    def fit(self, 
            X_train: pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
            X_test:  pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
            y_train: pl.DataFrame | pl.LazyFrame | pl.Series    | pd.Series | pd.DataFrame | np.ndarray, 
            y_test:  pl.DataFrame | pl.LazyFrame | pl.Series    | pd.Series | pd.DataFrame | np.ndarray
            ) ->     pd.DataFrame | np.ndarray:
        data = [X_train, X_test, y_train, y_test]
        
        for count, el in enumerate(data):
            data[count] = _to_numpy(el)

        models, predictions = self.clf.fit(data[0], data[1], data[2], data[3])
        return models, predictions

class RegressionGarden():
    def __init__(self,
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

    def fit(self, 
            X_train: pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
            X_test:  pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
            y_train: pl.DataFrame | pl.LazyFrame | pl.Series    | pd.Series | pd.DataFrame | np.ndarray, 
            y_test:  pl.DataFrame | pl.LazyFrame | pl.Series    | pd.Series | pd.DataFrame | np.ndarray
            ) ->     pd.DataFrame | np.ndarray:
        data = [X_train, X_test, y_train, y_test]

        for count, el in enumerate(data):
            data[count] = _to_numpy(el)
            
        models, predictions = self.reg.fit(data[0], data[1], data[2], data[3])
        return models, predictions
    
# ML UTILITY METRICS MODULE
def compute_utility_metrics_class( X_train:       pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
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
    ''' This function starts the training of a classification task on a pool of available classifiers and returns the metrics
    '''
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

def compute_utility_metrics_regr( X_train:       pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
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
    ''' This function starts the training of a regression task on a pool of available regressors and returns the metrics
    '''
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

def compute_statistical_metrics(real_data:    pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
                                synth_data:   pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray,
                                path_to_json: str = ""
                                ) -> Tuple[Dict, Dict, Dict]:
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

    # Drop columns that are present in the real dataset but not in the synthetic dataset
    real_data = _drop_real_cols(synth_data, real_data)

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

    # _save_to_json("real_data", real_data)
    # _save_to_json("synth_data", synth_data)

    return num_features_comparison, cat_features_comparison, time_features_comparison

def compute_mutual_info(real_data:  pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
                        synth_data: pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray,
                        path_to_json: str = ""):
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

    # Drop columns that are present in the real dataset but not in the synthetic dataset
    real_data = _drop_real_cols(synth_data, real_data)

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

    _save_to_json("real_corr", real_corr, path_to_json)
    _save_to_json("synth_corr", synth_corr, path_to_json)
    _save_to_json("diff_corr", diff_corr, path_to_json)

    return real_corr, synth_corr, diff_corr