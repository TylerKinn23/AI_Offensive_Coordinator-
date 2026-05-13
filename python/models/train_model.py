'''
This script is responsible for training the AI offensive coordinator using XGBoost. Model evaluation
and metrics will be done in later scripts.

This script will load processed data, define train/validation/test sets, tune hyperparameters, train the model,
and save all necessary artifacts.
'''

import pandas as pd
import numpy as np
import os
import yaml
import logging
import xgboost as xgb
import json
import joblib
from datetime import datetime
from ingestion.load_data import load_pipeline_config
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

def load_model_config():
    '''
    This function is responsible for loading in the model_config file. 
    '''

    # Get the directory that this script is in
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Moving up one level, then into the config folder
    config_path = os.path.join(base_dir, '..', 'configs', 'model_config.yaml')

    with open(config_path, 'r') as f:
        logging.info("Loading model config file")
        return yaml.safe_load(f)

def get_processed_data(pipeline_config):
    '''
    This function is responsible for retrieving the processed data from build_features.py
    and returning it as a pandas dataframe.
    '''

    # Get location of data
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Get name of data
    processed_folder = pipeline_config['data']['paths']['processed_data_path']
    df_name = pipeline_config['data']['file_naming']['processed_data_name']

    # Moving up one level, then into the data folder
    data_path = os.path.join(base_dir, '..', processed_folder, df_name)

    logging.info(f'Loading processed data from {data_path}')

    return pd.read_csv(data_path, low_memory = False)

def prepare_features(df, model_config):
    '''
    This function is responsible for defining the target variable and the rest of the features used for prediction given the config file.
    '''

    # Creating target variable and encoding it.
    y = df['play_subtype']
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    logging.info('Creating target feature')

    # Creating features to predict target variable.
    features = model_config['model_features']
    X = df[features]
    logging.info('Creating prediction features')

    return X, y, encoder

def split_data(X, y, pipeline_config):
    '''
    This function is responsible for splitting the data according to the user's specifications in the pipeline_config file.
    '''

    # Get random state
    seed = pipeline_config['data']['split_strategy']['random_seed']

    # Get sizes for each set
    test = pipeline_config['data']['split_strategy']['test_size']
    val = pipeline_config['data']['split_strategy']['validation_size']
    adjusted_val_size = val / (1 - test)
    if test + val >= 1:
        raise ValueError('The sum of the test and validation set size must be less than 1.')

    # Hold out test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test, random_state = seed)
    logging.info('Holding out test set')

    # Get train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = adjusted_val_size, random_state = seed)
    logging.info('Getting train and validation sets')

    return X_train, X_val, X_test, y_train, y_val, y_test

def tune_hyperparameters(model, model_config, X_train, y_train):
    '''
    This function is responsible for searching for optimal hyperparameters if the user has tuning enabled.
    It will return these optimal hyperparameters as a dictionary to be used in the model.
    '''

    # Create parameter distribution and other RandomizedSearchCV parameters
    param_dist = {
        'n_estimators': randint(model_config['model_params']['search_space']['n_estimators'][0], model_config['model_params']['search_space']['n_estimators'][1]),
        'max_depth': model_config['model_params']['search_space']['max_depth'],
        'learning_rate': model_config['model_params']['search_space']['learning_rate'],
        'subsample': model_config['model_params']['search_space']['subsample'],
        'colsample_bytree': model_config['model_params']['search_space']['colsample_bytree'],
        'lambda': model_config['model_params']['search_space']['lambda'],
        'alpha': model_config['model_params']['search_space']['alpha']
    }

    iterations = model_config['model_params']['tuning']['n_iter']
    cv_folds = model_config['model_params']['tuning']['cv_folds']
    jobs = model_config['model_params']['tuning']['n_jobs']
    
    search = RandomizedSearchCV(model, param_distributions = param_dist, n_iter = iterations, cv = cv_folds, n_jobs = jobs)
    search.fit(X_train, y_train)
    logging.info('Tuning hyperparameters')

    print(f"Best Parameters: {search.best_params_}")
    print(f"Best Accuracy: {search.best_score_}")

    return search.best_params_

def train_best_model(model, params, pipeline_config, X_train, y_train, X_val, y_val):
    '''
    This function will train the model based on "best" hyperparameters.
    '''

    # Get random state
    seed = pipeline_config['data']['split_strategy']['random_seed']

    trained_model = model(**params, random_state = seed)
    trained_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose = False)
    logging.info('Training model')

    return trained_model

def save_artifacts(model, encoder, params, feature_columns, X_test, y_test, model_config, pipeline_config):
    '''
    This function is responsible for saving all of the artifacts from the model training process within the
    trained_models folder under artifacts.
    '''

    # Get Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, '..', pipeline_config['data']['paths']['models'], pipeline_config['data']['file_naming']['models'])

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(save_path, f'xgb_model_{timestamp}')

    os.makedirs(run_dir, exist_ok = True)
    logging.info('Generating paths for model')

    # Defining new file paths
    model_path = os.path.join(run_dir, 'xgb_model.pkl')
    encoder_path = os.path.join(run_dir, 'label_encoder.pkl')
    params_path = os.path.join(run_dir, 'best_params.yaml')
    features_path = os.path.join(run_dir, 'feature_columns.json')
    X_test_path = os.path.join(run_dir, 'X_test_set.csv')
    y_test_path = os.path.join(run_dir, 'y_test_set.csv')
    metadata_path = os.path.join(run_dir, 'training_metadata.json')

    # Begin saving process
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)

    with open(params_path, 'w') as f:
        yaml.dump(params, f)

    feature_list = list(feature_columns.columns)
    with open(features_path, 'w') as f:
        json.dump(feature_list, f, indent=4)

    X_test.to_csv(X_test_path, index = False)
    y_test.to_csv(y_test_path, index = False)

    # Create and save model metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': 'XGB' + str(model_config['model_params']['objective']),
        'num_features': len(feature_list),
        'test_size': pipeline_config['data']['split_strategy']['test_size'],
        'validation_size': pipeline_config['data']['split_strategy']['validation_size'],
        'random_seed': pipeline_config['data']['split_strategy']['random_seed']
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    logging.info(f"Artifacts saved successfully to {run_dir}")

    return run_dir

def main():

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    model = xgb.XGBClassifier

    try:
        model_config = load_model_config()
        pipeline_config = load_pipeline_config()

        data = get_processed_data(pipeline_config)

        X, y, encoder = prepare_features(data, model_config)

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, pipeline_config)

        if model_config['model_params']['tuning']['enabled']:
            params = tune_hyperparameters(model, model_config, X_train, y_train)
        else:
            params = {
                'n_estimators': model_config['model_params']['n_estimators'], 
                'max_depth': model_config['model_params']['max_depth'],
                'learning_rate': model_config['model_params']['learning_rate'],
                'subsample': model_config['model_params']['subsample'],
                'colsample_bytree': model_config['model_params']['colsample_bytree'],
                'lambda': model_config['model_params']['lambda'],
                'alpha': model_config['model_params']['alpha']
            }

        model = train_best_model(model, params, pipeline_config, X_train, y_train, X_val, y_val)

    except FileNotFoundError as e:
        logging.error(f"CONFIG ERROR: Could not find YAML file. Details: {e}")
        return 1
    
    except ValueError as e:
        logging.error(f"PARAMETER ERROR: The given parameters failed integrity checks. Details: {e}")
        return 1
    
    except Exception as e:
        logging.error(f"UNEXPECTED ERROR: An unknown error occurred: {e}")
        return 1
    
    else:
        save_artifacts(model, encoder, params, X.columns, X_test, y_test, model_config, pipeline_config)
        return 0
    
if __name__ == "__main__":
    main()
