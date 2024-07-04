from typing import Callable, List

import numpy  as np
import pandas as pd
import polars as pl

def compute_utility_metrics_class( X_train:       pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
                                   X_test:        pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
                                   y_train:       pl.DataFrame | pl.LazyFrame | pl.Series    | pd.Series | pd.DataFrame | np.ndarray,  
                                   y_test:        pl.DataFrame | pl.LazyFrame | pl.Series    | pd.Series | pd.DataFrame | np.ndarray, 
                                   custom_metric: Callable       = None, 
                                   classifiers:   List[Callable] = "all",
                                   predictions:   bool = False
                                 ): 
    from metrics_modules.model_garden import ClassificationSandbox

    classifier = ClassificationSandbox(predictions=predictions, classifiers=classifiers, custom_metric=custom_metric)
    models, pred = classifier.fit(X_train, X_test, y_train, y_test)
    return models, pred

def compute_utility_metrics_regr( X_train:        pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
                                  X_test:         pl.DataFrame | pl.LazyFrame | pd.DataFrame | np.ndarray, 
                                  y_train:        pl.DataFrame | pl.LazyFrame | pl.Series    | pd.Series | pd.DataFrame | np.ndarray,  
                                  y_test:         pl.DataFrame | pl.LazyFrame | pl.Series    | pd.Series | pd.DataFrame | np.ndarray, 
                                  custom_metric:  Callable       = None, 
                                  regressors:     List[Callable] = "all",
                                  predictions:    bool = False
                                ):
    from metrics_modules.model_garden import RegressionSandbox

    regressor = RegressionSandbox(predictions=predictions, classifiers=regressors, custom_metric=custom_metric)
    models, pred = regressor.fit(X_train, X_test, y_train, y_test)
    return models, pred