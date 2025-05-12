import pytest
import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
from sure import Preprocessor
from sure.utility import (compute_statistical_metrics, compute_mutual_info,
			  compute_utility_metrics_class)
from sure.privacy import (distance_to_closest_record, dcr_stats, number_of_dcr_equal_to_zero, validation_dcr_test, 
			  adversary_dataset, membership_inference_test)
@pytest.fixture
def real_data():
    test_dir = Path(__file__).parent
    df = pl.read_csv(test_dir / "resources" / "dataset.csv")
    return df

@pytest.fixture
def synthetic_data():
    test_dir = Path(__file__).parent
    df = pl.read_csv(test_dir / "resources" / "synthetic_dataset.csv")
    return df

@pytest.fixture
def validation_data():
    test_dir = Path(__file__).parent
    df = pl.read_csv(test_dir / "resources" / "validation_dataset.csv")
    return df

def test_statistical_metrics(real_data, synthetic_data):
    """Test computation of statistical metrics between real and synthetic data"""    
    num_stats, cat_stats, _ = compute_statistical_metrics(real_data, synthetic_data)
    
    assert isinstance(num_stats, dict)

def test_mutual_info(real_data, synthetic_data):
    """Test computation of mutual information metrics"""
    preprocessor = Preprocessor(real_data, num_fill_null='forward', scaling='standardize')
    real_preprocessed = preprocessor.transform(real_data)
    synth_preprocessed = preprocessor.transform(synthetic_data)
    
    corr_real, corr_synth, corr_diff = compute_mutual_info(real_preprocessed, synth_preprocessed)
    
    assert corr_real.shape == corr_synth.shape

def test_privacy_metrics(real_data, synthetic_data, validation_data):
    """Test computation of privacy metrics"""    
    # Test DCR computations
    dcr_synth_train = distance_to_closest_record("synth_train", synthetic_data, real_data)
    dcr_synth_valid = distance_to_closest_record("synth_val", synthetic_data, validation_data)
        
    # Test DCR stats
    stats_train = dcr_stats("synth_train", dcr_synth_train)
    stats_valid = dcr_stats("synth_val", dcr_synth_valid)
        
    # Test validation
    share = validation_dcr_test(dcr_synth_train, dcr_synth_valid)
    assert isinstance(share['percentage'], float)
    assert 0 <= share['percentage'] <= 100

def test_membership_inference(real_data, synthetic_data, validation_data):
    """Test membership inference attack"""    
    adversary_df = adversary_dataset(real_data, validation_data)
    ground_truth = adversary_df["privacy_test_is_training"]
    
    mia_results = membership_inference_test(adversary_df, synthetic_data, ground_truth)
    
    assert isinstance(mia_results, dict)
