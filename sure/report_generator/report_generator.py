import subprocess
import pkg_resources
import sys

import json
import os
import tempfile

import pandas as pd
import polars as pl
import numpy as np

# Function to run the streamlit app
def report(df_real:  pd.DataFrame | pl.DataFrame | pl.LazyFrame, 
           df_synth: pd.DataFrame | pl.DataFrame | pl.LazyFrame, 
           path_to_json:str = ""):
    """Generate the report app"""
    # Check dataframe type
    if isinstance(df_real, pd.DataFrame) and isinstance(df_synth, pd.DataFrame):
        pass
    elif isinstance(df_real, pl.DataFrame) and isinstance(df_synth, pl.DataFrame):
        df_real  = df_real.to_pandas()
        df_synth = df_synth.to_pandas()
    elif isinstance(df_real, pl.LazyFrame) and isinstance(df_synth, pl.LazyFrame):
        df_real  = df_real.collect().to_pandas()
        df_synth = df_synth.collect().to_pandas()
    else:
        sys.exit('ErrorType\nThe datatype provided is not supported or the two datasets have different types.')

    # Save the DataFrame to a temporary file (pickle format)
    # if df_real:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmpfile:
        df_path_real = tmpfile.name
        df_real.to_pickle(df_path_real)  # Save the DataFrame as a pickle file
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmpfile:
        df_path_synth = tmpfile.name
        df_synth.to_pickle(df_path_synth)  # Save the DataFrame as a pickle file

    report_path = pkg_resources.resource_filename('sure.report_generator', 'report_app.py')
    process = subprocess.run(['streamlit', 'run', report_path, df_path_real, df_path_synth, path_to_json])
    return process

def _convert_to_serializable(obj: object):
    """Recursively convert DataFrames and other non-serializable objects in a nested dictionary to serializable formats"""
    if isinstance(obj, (pd.DataFrame, pl.DataFrame, pl.LazyFrame)):
        if isinstance(obj, pl.DataFrame):
            obj = obj.to_pandas()
        if isinstance(obj, pl.LazyFrame):
            obj = obj.collect().to_pandas()
        
        # Convert index to column only if index is non-numerical
        if obj.index.dtype == 'object' or pd.api.types.is_string_dtype(obj.index):
            obj = obj.reset_index()
        
        # Convert datetime columns to string
        for col in obj.columns:
            if pd.api.types.is_datetime64_any_dtype(obj[col]):
                obj[col] = obj[col].astype(str)
        
        return obj.to_dict(orient='records')
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def _save_to_json(data_name: str, 
                  new_data: object,
                  path_to_json: str):
    """Save data into a JSON file in the folder where the user is working"""
    # Check if the file exists
    path = os.path.join(path_to_json,"data.json")
    if os.path.exists(path):
        # Read the existing data from the file
        with open(path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}  # Initialize to an empty dictionary if the file is empty or invalid
    else:
        data = {}

    # Convert new_data to a serializable format if it is a DataFrame or contains DataFrames
    serializable_data = _convert_to_serializable(new_data)

    # Update the data dictionary with the new_data
    data[data_name] = serializable_data

    # Write the updated data back to the file
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

def _load_from_json(path_to_json: str,
                    data_name: str = None
                    ):
    """Load data from a JSON file "data.json" in the folder where the user is working"""
    # Check if the file exists
    path = os.path.join(path_to_json,"data.json")
    if not os.path.exists(path):
        raise FileNotFoundError("The data.json file does not exist.")
    
    # Read the data from the file
    with open(path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            raise ValueError("The data.json file is empty or invalid.")

    if data_name:
        # Extract the relevant data
        if data_name not in data:
            raise KeyError(f"Key '{data_name}' not found in dictionary.")
        data = data.get(data_name, None)
    
    return data

def _convert_to_dataframe(obj):
    """Convert nested dictionaries back to DataFrames"""
    if isinstance(obj, list) and all(isinstance(item, dict) for item in obj):
        return pd.DataFrame(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_dataframe(v) for k, v in obj.items()}
    else:
        return obj