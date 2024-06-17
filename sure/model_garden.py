import numpy  as np
import polars as pl
import pandas as pd

from lazypredict.Supervised import LazyClassifier, LazyRegressor

class ClassificationSandbox:

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
        for el in data:
            if isinstance(el, pl.LazyFrame):
                data[el] = el.collect().to_numpy()
            elif isinstance(el, pl.DataFrame | pd.DataFrame| pl.Series | pd.Series):
                data[el] = el.to_numpy()
            elif isinstance(el, np.ndarray):
                pass
            else:
                print("The dataframe must be a Polars LazyFrame or a Pandas DataFrame")
                return
        
        models, predictions = self.clf.fit(data[0], data[1], data[2], data[3])
        return models, predictions

class RegressionSandbox():

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
        for el in data:
            if isinstance(el, pl.LazyFrame):
                data[el] = el.collect().to_numpy()
            elif isinstance(el, pl.DataFrame | pd.DataFrame| pl.Series | pd.Series):
                data[el] = el.to_numpy()
            else:
                print("The dataframe must be a Polars LazyFrame, a Pandas DataFrame or a Numpy Array")
                return
            
        models, predictions = self.reg.fit(data[0], data[1], data[2], data[3])
        return models, predictions