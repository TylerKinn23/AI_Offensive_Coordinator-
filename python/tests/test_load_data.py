import pytest
import textwrap
import pandas as pd
import os
from ingestion.load_data import load_config, validate_data, save_raw_data
from unittest.mock import patch, mock_open

def test_load_config_data():
    '''
    This function is responsible for testing if data is loaded into the config.
    '''

    # Simulated YAML file
    fake_yaml = textwrap.dedent('''
    data:
      seasons: [2022, 2023]
      paths:
        raw_data_path: "data/raw"
    ''')

    # Ensuring that the fake yaml file is opened instead of the real one
    with patch('builtins.open', mock_open(read_data = fake_yaml)):
        config = load_config()

    # Checking function with assertions
    assert 'data' in config
    assert config['data']['seasons'] == [2022, 2023]

def test_load_config_instance():
    '''
    This function is responsible for testing if the config is instantiated as a dictionary.
    '''

    # Simulated YAML file
    fake_yaml = textwrap.dedent('''
    data:
      seasons: [2022, 2023]
      paths:
        raw_data_path: "data/raw"
    ''')

    # Ensuring that the fake yaml file is opened instead of the real one
    with patch('builtins.open', mock_open(read_data = fake_yaml)):
        config = load_config()

    # Checking function with assertions
    assert isinstance(config, dict)

def test_load_config_path():
    '''
    This function is responsible for testing if the config has a correct path.
    '''

    # Simulated YAML file
    fake_yaml = textwrap.dedent('''
    data:
      seasons: [2022, 2023]
      paths:
        raw_data_path: "data/raw"
    ''')

    # Ensuring that the fake yaml file is opened instead of the real one
    with patch('builtins.open', mock_open(read_data = fake_yaml)):
        config = load_config()

    # Checking function with assertions
    assert config['data']['paths']['raw_data_path'] == "data/raw"

def test_validate_data_incorrect(): 
    '''
    This function is responsible for testing if the validate_data function can detect incorrect columns.
    '''

    # Simulating config file and invalid data
    invalid_data = pd.DataFrame({'incorrect_column': [1,2,3]})
    config = {
    'features': {
        'required_raw_nflverse': ['play_type'],
        'historical_features': {'enabled': False}  # Added this to satisfy the function
    }
    }   

    # Checking with assertions
    with pytest.raises(KeyError):
        validate_data(invalid_data, config)

def test_validate_data_empty(): 
    '''
    This function is responsible for testing if the validate_data function can detect an empty dataframe.
    '''

    # Simulating config file and invalid data
    empty_data = pd.DataFrame()
    config = {
    'features': {
        'required_raw_nflverse': ['play_type'],
        'historical_features': {'enabled': False}  # Added this to satisfy the function
    }
    }   

    # Checking with assertions
    with pytest.raises(ValueError):
        validate_data(empty_data, config)

def test_validate_data_none(): 
    '''
    This function is responsible for testing if the validate_data function can detect an None being passed to it.
    '''

    # Simulating config file 
    config = {
    'features': {
        'required_raw_nflverse': ['play_type'],
        'historical_features': {'enabled': False}  # Added this to satisfy the function
    }
    }   

    # Checking with assertions
    with pytest.raises(ValueError):
        validate_data(None, config)

def test_validate_data_correct(): 
    '''
    This function is responsible for testing if the validate_data function will return true when a valid dataframe is passed onto it.
    '''

    # Simulating config file and data
    valid_data = pd.DataFrame({'play_type': ['pass', 'run'], 'run_location': 'middle', 'pass_length': 0})
    config = {
    'features': {
        'required_raw_nflverse': ['play_type'],
        'historical_features': {'enabled': False}  # Added this to satisfy the function
    }
    }   

    # Checking with assertions
    assert validate_data(valid_data, config) is True
    
def test_save_raw_data_creates_dir(tmp_path):
    '''
    This function is responsible for testing if the save_raw_data function will auto-create a directory if it doesn't exist.
    '''

    # Creating dummy dataframe
    df = pd.DataFrame({'test': [1,2,3]})

    # Define nested path that does not exist
    fake_raw_dir = 'test_data/new_raw_folder'
    file_name = 'test.csv'

    config = {
        'data': {
            'paths': {'raw_data_path': str(tmp_path / fake_raw_dir)},
            'file_naming': {'raw_data_name': file_name}
        }
    }

    # Activating function
    save_raw_data(df, config)

    # Checking function behavior with assertions
    expected_dir = tmp_path / fake_raw_dir
    assert os.path.exists(expected_dir)
    assert os.path.isdir(expected_dir)

def test_save_raw_data_creates_file(tmp_path):
    '''
    This function is responsible for testing if the save_raw_data function will create a file.
    '''
    
    # Creating dummy data/paths
    sample_data = pd.DataFrame({'play_id': [101, 102], 'play_type': ['pass', 'run']})
    file_name = 'test.csv'

    save_dir = str(tmp_path / 'data')

    config = {
        'data': {
            'paths': {'raw_data_path': save_dir},
            'file_naming': {'raw_data_name': file_name}
        }
    }

    # Activating Function
    save_raw_data(sample_data, config)

    # Checking existence and correctness of saved file
    file_path = tmp_path / 'data' / file_name
    assert os.path.exists(file_path)
    
    saved_df = pd.read_csv(file_path)
    assert saved_df.shape == (2,2)
    

