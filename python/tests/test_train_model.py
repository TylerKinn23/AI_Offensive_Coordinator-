'''
This function is responsible for testing the train_model.py script.
'''

import pytest
import textwrap
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, mock_open
from models.train_model import load_model_config, get_processed_data, prepare_features, split_data, tune_hyperparameters, train_best_model, save_artifacts

def test_load_model_config_data():
    '''
    This function is responsible for testing if data is loaded into the config.
    '''

    # Simulated YAML file
    fake_yaml = textwrap.dedent('''
    data:
      target: "play_subtype"
    model_params:
      num_class: 4
    ''')

    # Ensuring that the fake yaml file is opened instead of the real one
    with patch('builtins.open', mock_open(read_data = fake_yaml)):
        config = load_model_config()

    # Checking function with assertions
    assert 'data' in config
    assert config['data']['target'] == "play_subtype"

def test_load_model_config_instance():
    '''
    This function is responsible for testing if the config is instantiated as a dictionary.
    '''

    # Simulated YAML file
    fake_yaml = textwrap.dedent('''
    data:
      target: "play_subtype"
    model_params:
      num_class: 4
    ''')

    # Ensuring that the fake yaml file is opened instead of the real one
    with patch('builtins.open', mock_open(read_data = fake_yaml)):
        config = load_model_config()

    # Checking function with assertions
    assert isinstance(config, dict)

def test_get_processed_data_path():
    '''
    This function will test the ability of get_processed_data to properly get paths/filenames from the config file it is passed.
    '''

    # Create fake config file.
    fake_config = {
        'data': {
            'paths': {'processed_data_path': 'fake/processed'},
            'file_naming': {'processed_data_name': 'test_data.csv'}
        }
    }

    # Use mocking to open the fake file
    with patch('pandas.read_csv') as mock_read:
        get_processed_data(fake_config)

        called_path = mock_read.call_args[0][0]

    assert "fake/processed" in called_path
    assert "test_data.csv" in called_path

def test_get_cleaned_data_FileNotFoundError():
    '''
    This function will test if get_processed_data properly raises an error when a file is not found.
    '''

    # Create fake config (with a nonexistent path)
    fake_config = {
        'data': {
            'paths': {'processed_data_path': 'does/not'},
            'file_naming': {'processed_data_name': 'exist.csv'}
        }
    }

    with pytest.raises(FileNotFoundError):
        get_processed_data(fake_config)

def test_prepare_features_structure():
    '''
    This function will test that the prepare_features function properly maintains data structure when separating X and y variables as well as encoding.
    '''

    # Create fake test data
    df = pd.DataFrame({
        'play_subtype': ['pass', 'run', 'pass'],
        'down': [1, 2, 3],
        'dist': [10, 5, 2]
    })
    model_config = {'model_features': ['down', 'dist']}
    
    X, y, encoder = prepare_features(df, model_config)
    
    assert X.shape == (3, 2)
    assert list(X.columns) == ['down', 'dist']
    assert len(np.unique(y)) == 2

def test_prepare_features_encoding():
    '''
    This function will test that the prepare_features function properly encodes the y variable as it is categorical.
    '''

    # Create fake test data
    df = pd.DataFrame({
        'play_subtype': ['pass', 'run', 'pass'],
        'down': [1, 2, 3],
        'dist': [10, 5, 2]
    })
    model_config = {'model_features': ['down', 'dist']}
    
    X, y, encoder = prepare_features(df, model_config)
    assert isinstance(y[0], (int, np.integer))

def test_split_data_math():
    '''
    This function will test if the split_data funciton properly splits the data given user config input.
    '''

    # Create test data/config
    X = pd.DataFrame(np.random.rand(100, 2))
    y = pd.Series(np.random.randint(0, 2, 100))
    pipeline_config = {
        'data': {
            'split_strategy': {
                'test_size': 0.2, # 20 rows
                'validation_size': 0.1, # 10 rows
                'random_seed': 42
            }
        }
    }
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, pipeline_config)
    
    assert len(X_test) == 20
    assert len(X_val) == 10
    assert len(X_train) == 70

def test_split_data_ValueError():
    '''
    This function will test if the split_data funciton properly raises a ValueError for incorrect user input.
    '''

    # Create test data/config
    X = pd.DataFrame(np.random.rand(100, 2))
    y = pd.Series(np.random.randint(0, 2, 100))
    pipeline_config = {
        'data': {
            'split_strategy': {
                'test_size': 0.5, 
                'validation_size': 0.5, 
                'random_seed': 42
            }
        }
    }
    
    with pytest.raises(ValueError):
        split_data(X, y, pipeline_config)

@patch('models.train_model.RandomizedSearchCV')
def test_tune_hyperparameters_logic(mock_search):
    '''
    This function will ensure that tune hyperparameters is passing the right parameters to be searched for to RandomizedSearchCV
    '''
    
    # Creating Mocks of objects so that the actual funciton isn't called.
    mock_instance = mock_search.return_value
    mock_instance.best_params_ = {'max_depth': 5}
    mock_instance.best_score_ = 0.85
    
    model_config = {
        'model_params': {
            'search_space': {
                'n_estimators': [10, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'lambda': [1],
                'alpha': [0]
            },
            'tuning': {'n_iter': 5, 'cv_folds': 3, 'n_jobs': -1}
        }
    }

    params = tune_hyperparameters(MagicMock(), model_config, MagicMock(), MagicMock())
    
    args, kwargs = mock_search.call_args
    assert kwargs['n_iter'] == 5
    assert kwargs['cv'] == 3
    assert params == {'max_depth': 5}

def test_train_best_model_calls_fit():
    '''
    This function will test if fit was called when training the best model. 
    '''

    # Mock the XGB Class
    mock_model_class = MagicMock()
    mock_instance = mock_model_class.return_value
    
    params = {'n_estimators': 10}
    config = {'data': {'split_strategy': {'random_seed': 42}}}
    X, y = MagicMock(), MagicMock()

    train_best_model(mock_model_class, params, config, X, y, X, y)

    # Check if fit was called
    assert mock_instance.fit.called

def test_train_best_model_eval_set():
    '''
    This function will test if the evaluation set was passed to the train_best_model function.
    '''

    # Mock the XGB Class
    mock_model_class = MagicMock()
    mock_instance = mock_model_class.return_value
    
    params = {'n_estimators': 10}
    config = {'data': {'split_strategy': {'random_seed': 42}}}
    X, y = MagicMock(), MagicMock()

    train_best_model(mock_model_class, params, config, X, y, X, y)

    # Check if eval_set was passed (for your validation set logic)
    args, kwargs = mock_instance.fit.call_args
    assert 'eval_set' in kwargs

@patch('pandas.Series.to_csv')
@patch('pandas.DataFrame.to_csv')
@patch('builtins.open', new_callable=mock_open)
@patch('joblib.dump')
@patch('os.makedirs')
def test_save_artifacts_logic(mock_make, mock_dump, mock_file, mock_df_csv, mock_ser_csv):
    # Setup minimal mocks for the arguments
    model = MagicMock()
    encoder = MagicMock()
    X_test = pd.DataFrame({'a': [1]})
    y_test = np.array([1])
    
    # We also need to mock the configs so the function doesn't crash 
    # trying to look up keys like ['data']['paths']['models']
    mock_model_config = {
        'model_params': {'objective': 'binary'}
    }
    mock_pipeline_config = {
        'data': {
            'paths': {'models': 'fake_models'},
            'file_naming': {'models': 'fake_name'},
            'split_strategy': {'test_size': 0.2, 'validation_size': 0.1, 'random_seed': 42}
        },
        'model_settings': {'model_name': 'test_model'}
    }
    
    # Execute the function
    save_artifacts(
        model, encoder, {}, X_test.columns, X_test, y_test, 
        mock_model_config, mock_pipeline_config
    )
    
    # Assertions
    assert mock_make.called
    assert mock_dump.call_count >= 2  # Model and Encoder
    assert mock_df_csv.called        # X_test save
    assert mock_ser_csv.called       # y_test save
    assert mock_file.called          # JSON metadata/features save