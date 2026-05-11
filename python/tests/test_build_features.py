'''
This function is responsible for testing the build_features.py script.
'''

import pytest
import textwrap
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
from features.build_features import get_cleaned_data, create_historical_features, save_processed_data

# Creating all necessary pytest fixtures to test create_historical_features
@pytest.fixture
def mock_config():
    """Provides a controlled configuration for feature engineering tests."""
    return {
        'features': {
            'historical_features': {
                'enabled': True,
                'rolling_windows': 3
            }
        }
    }

@pytest.fixture
def seasonal_data():
    """
    Data spanning two seasons for the same team to test boundary isolation.
    KC 2022: Weeks 17, 18
    KC 2023: Week 1
    """
    return pd.DataFrame({
        'defteam': ['KC'] * 3,
        'season': [2022, 2022, 2023],
        'week': [17, 18, 1],
        'yards_gained': [100, 100, 500],
        'play_type': ['run'] * 3
    })

@pytest.fixture
def multi_team_data():
    """
    Data for two different teams to ensure stats don't leak across teams.
    """
    return pd.DataFrame({
        'defteam': ['KC', 'KC', 'PHI', 'PHI'],
        'season': [2023, 2023, 2023, 2023],
        'week': [1, 2, 1, 2],
        'yards_gained': [100, 200, 50, 50],
        'play_type': ['run'] * 4
    })

@pytest.fixture
def rolling_math_data():
    """
    Standard 4-week dataset for one team.
    Expected Raw Means:
    Week 1: 100
    Week 2: 200
    Week 3: 300
    Week 4: 400
    """
    return pd.DataFrame({
        'defteam': ['KC'] * 4,
        'season': [2023] * 4,
        'week': [1, 2, 3, 4],
        'yards_gained': [100, 200, 300, 400],
        'play_type': ['run'] * 4
    })

@pytest.fixture
def rush_vs_pass_data():
    """
    Data where a week contains both runs and passes.
    Week 1: 
       - 100 yard Run
       - 300 yard Pass
    Expected behavior: 
       The 'avg_rush_yds_allowed_play' for Week 1 should be 100.
       Therefore, Week 2's rolling feature should be 100 (ignoring the 300).
    """
    return pd.DataFrame({
        'defteam': ['KC', 'KC', 'KC'],
        'season': [2023, 2023, 2023],
        'week': [1, 1, 2],
        'yards_gained': [100, 300, 500],
        'play_type': ['run', 'pass', 'run'] # One of each in Week 1
    })

def test_get_cleaned_data_path():
    '''
    This function will test the ability of get_cleaned_data to properly get paths/filenames from the config file it is passed.
    '''

    # Create fake config file.
    fake_config = {
        'data': {
            'paths': {'cleaned_data_path': 'fake/cleaned'},
            'file_naming': {'cleaned_data_name': 'test_data.csv'}
        }
    }

    # Use mocking to open the fake file
    with patch('pandas.read_csv') as mock_read:
        get_cleaned_data(fake_config)

        called_path = mock_read.call_args[0][0]

    assert "fake/cleaned" in called_path
    assert "test_data.csv" in called_path

def test_get_cleaned_data_FileNotFoundError():
    '''
    This function will test if get_cleaned_data properly raises an error when a file is not found.
    '''

    # Create fake config (with a nonexistent path)
    fake_config = {
        'data': {
            'paths': {'cleaned_data_path': 'does/not'},
            'file_naming': {'cleaned_data_name': 'exist.csv'}
        }
    }

    with pytest.raises(FileNotFoundError):
        get_cleaned_data(fake_config)

def test_create_historical_features_value_error():
    '''
    This function will test if create_historical_features properly raises a value error for empty/null data after filtering.
    '''
    
    # Defining dummy config
    config = {}

    with pytest.raises(ValueError):
        create_historical_features(pd.DataFrame(), config)

    with pytest.raises(ValueError):
        create_historical_features(None, config)

def test_season_boundary_isolation(seasonal_data, mock_config):
    '''
    This function will ensure that data from a previous season is not leaking into next when calculating historical features.
    '''

    # Get testing dataframe
    test_df = create_historical_features(seasonal_data, mock_config)

    invalid = test_df[(test_df['week'] == 1)]
    assert len(invalid) == 0

def test_rolling_math_accuracy(rolling_math_data, mock_config):
    '''
    This function will verify the accuracy of rolling mean calculations.
    '''

    # Get testing dataframe
    test_df = create_historical_features(rolling_math_data, mock_config)

    week2 = test_df[(test_df['week'] == 2)]
    assert week2['avg_yds_pp_alwd_rolling'].item() == 100
    week3 = test_df[(test_df['week'] == 3)]
    assert week3['avg_yds_pp_alwd_rolling'].item() == 150
    week4 = test_df[(test_df['week'] == 4)]
    assert week4['avg_yds_pp_alwd_rolling'].item() == 200

def test_rush_math_ignores_passes(rush_vs_pass_data, mock_config):
    '''
    This function is responsible for ensuring the avg_rush_yds_per_rush_alwd_rolling column ignores pass plays in its calculations.
    '''

    # Get testing dataframe
    test_df = create_historical_features(rush_vs_pass_data, mock_config)

    week2 = test_df[(test_df['week'] == 2)]
    assert week2['avg_rush_yds_per_rush_alwd_rolling'].item() == 100

def test_team_isolation(multi_team_data, mock_config):
    '''
    This function ensures that team rolling stats are calculated in isolation and that teams are not impacting other team's stats.
    '''

    # Get testing dataframe
    test_df = create_historical_features(multi_team_data, mock_config)

    week2_phi = test_df[(test_df['week'] == 2) & (test_df['defteam'] == 'PHI')]
    assert week2_phi['avg_yds_pp_alwd_rolling'].item() == 50
    week2_KC = test_df[(test_df['week'] == 2) & (test_df['defteam'] == 'KC')]
    assert week2_KC['avg_yds_pp_alwd_rolling'].item() == 100
