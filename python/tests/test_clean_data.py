'''
This function is responsible for testing the clean_data.py script.
'''

import pytest
import textwrap
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
from preprocessing.clean_data import get_raw_data, initial_clean, classify_play, filter_rows, save_cleaned_data

@pytest.fixture
def sample_pbp_data():
    '''
    This is a representative NFLVerse dataframe which includes some noise that the clean_data functions should
    handle. It will be used for testing
    '''

    return pd.DataFrame({
        'play_type': ['run', 'run', 'pass', 'pass', 'run', 'pass', 'sack'],
        'run_location': ['middle', 'left', np.nan, np.nan, np.nan, 'right', np.nan],
        'pass_length': [np.nan, np.nan, 'short', 'deep', np.nan, np.nan, np.nan],
        'special': [0, 0, 0, 0, 0, 0, 0],
        'two_point_attempt': [0, 0, 0, 0, 0, 0, 0],
        'sack': [0, 0, 0, 0, 0, 0, 1],
        'qb_scramble': [0, 0, 0, 0, 0, 0, 0],
        'qb_kneel': [0, 0, 0, 0, 0, 0, 0],
        'qb_spike': [0, 0, 0, 0, 0, 0, 0]
    })

@pytest.fixture
def clean_sample_data():
    '''
    Provides a dataset that has already been through initial_clean.
    Contains exactly 2 of each subtype to be engineered.
    '''
    return pd.DataFrame({
        'play_type': [
            'run', 'run',    # inside
            'run', 'run',    # outside
            'pass', 'pass',  # short
            'pass', 'pass',  # deep
            'run'            # other
        ],
        'run_location': [
            'middle', 'middle', 
            'left', 'right', 
            np.nan, np.nan, 
            np.nan, np.nan, 
            np.nan           # The "other" run
        ],
        'pass_length': [
            np.nan, np.nan, 
            np.nan, np.nan, 
            'short', 'short', 
            'deep', 'deep', 
            np.nan           # The "other" pass/run
        ]
    })

def test_get_raw_data_path():
    '''
    This function will test the ability of get_raw data to properly get paths/filenames from the config file it is passed.
    '''

    # Create fake config file.
    fake_config = {
        'data': {
            'paths': {'raw_data_path': 'fake/raw'},
            'file_naming': {'raw_data_name': 'test_data.csv'}
        }
    }

    # Use mocking to open the fake file
    with patch('pandas.read_csv') as mock_read:
        get_raw_data(fake_config)

        called_path = mock_read.call_args[0][0]

    assert "fake/raw" in called_path
    assert "test_data.csv" in called_path

def test_get_raw_data_FileNotFoundError():
    '''
    This function will test if get_raw_data properly raises an error when a file is not found.
    '''

    # Create fake config (with a nonexistent path)
    fake_config = {
        'data': {
            'paths': {'raw_data_path': 'does/not'},
            'file_naming': {'raw_data_name': 'exist.csv'}
        }
    }

    with pytest.raises(FileNotFoundError):
        get_raw_data(fake_config)

def test_inital_clean_basic_filter(sample_pbp_data):
    '''
    This function is responsible for testing if initial_clean is able to properly do basic filtering.
    '''

    # Getting the filtered version of the above sample data.
    test_df = initial_clean(sample_pbp_data)

    # Checking basic filtering capability using assertions.
    assert set(test_df['play_type'].unique()).issubset({'run', 'pass'})
    noise = ['special', 'two_point_attempt', 'sack', 'qb_scramble', 'qb_kneel', 'qb_spike']
    assert (test_df[noise] == 0).all().all()

def test_initial_clean_validation(sample_pbp_data):
    '''
    This function will test if initial_clean is able to remove run plays with missing run location and pass plays with missing pass length.
    '''

    # Getting filtered version of sample data.
    test_df = initial_clean(sample_pbp_data)

    invalid_runs = test_df[(test_df['play_type'] == 'run') & (test_df['run_location'].isna())]
    assert len(invalid_runs) == 0

    valid_passes = test_df[(test_df['play_type'] == 'pass') & (test_df['pass_length'].notna())]
    assert len(valid_passes) > 0

def test_classify_play_classification(clean_sample_data):
    '''
    This function will test if classify_play can correctly classify plays including other.
    '''

    # Getting classified sample data.
    test_df = classify_play(clean_sample_data)

    assert 'play_subtype' in test_df.columns
    counts = test_df['play_subtype'].value_counts()
    assert counts['inside run'] == 2
    assert counts['outside run'] == 2
    assert counts['short pass'] == 2
    assert counts['deep pass'] == 2
    assert counts['other'] == 1
    assert (test_df[test_df['run_location'] == 'middle']['play_subtype'] == 'inside run').all()

def test_filter_rows_removal(clean_sample_data):
    '''
    This function will test the ability of filter_rows to remove columns with 'play_subtype' = other.
    '''

    # Getting dataframe
    dummy = classify_play(clean_sample_data)
    test_df = filter_rows(dummy)

    invalid = test_df[(test_df['play_subtype'] == 'other')]
    assert len(invalid) == 0

def test_filter_rows_value_error():
    '''
    This function will test if filter_rows properly raises a value error for empty/null data after filtering.
    '''

    # Define only other DF
    all_other = pd.DataFrame({'play_subtype': ['other', 'other']})

    with pytest.raises(ValueError):
        filter_rows(all_other)

    with pytest.raises(ValueError):
        filter_rows(pd.DataFrame())

    with pytest.raises(ValueError):
        filter_rows(None)

def test_save_cleaned_data_creates_dir(tmp_path):
    '''
    This function is responsible for testing if the save_cleaned_data function will auto-create a directory if it doesn't exist.
    '''

    # Creating dummy dataframe
    df = pd.DataFrame({'test': [1,2,3]})

    # Define nested path that does not exist
    fake_cleaned_dir = 'test_data/new_cleaned_folder'
    file_name = 'test.csv'

    config = {
        'data': {
            'paths': {'cleaned_data_path': str(tmp_path / fake_cleaned_dir)},
            'file_naming': {'cleaned_data_name': file_name}
        }
    }

    # Activating function
    save_cleaned_data(df, config)

    # Checking function behavior with assertions
    expected_dir = tmp_path / fake_cleaned_dir
    assert os.path.exists(expected_dir)
    assert os.path.isdir(expected_dir)

def test_save_cleaned_data_creates_file(tmp_path):
    '''
    This function is responsible for testing if the save_cleaned_data function will create a file.
    '''
    
    # Creating dummy data/paths
    sample_data = pd.DataFrame({'play_id': [101, 102], 'play_type': ['pass', 'run']})
    file_name = 'test.csv'

    save_dir = str(tmp_path / 'data')

    config = {
        'data': {
            'paths': {'cleaned_data_path': save_dir},
            'file_naming': {'cleaned_data_name': file_name}
        }
    }

    # Activating Function
    save_cleaned_data(sample_data, config)

    # Checking existence and correctness of saved file
    file_path = tmp_path / 'data' / file_name
    assert os.path.exists(file_path)
    
    saved_df = pd.read_csv(file_path)
    assert saved_df.shape == (2,2)
