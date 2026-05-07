'''
This program is responsible for reliably aquiring raw data from NFLVerse.
It is also responsible for validating the data and storing the raw dataset.
Preprocessing steps will take place in clean_data.py
'''

import nflreadpy as nfl
import logging
import pandas as pd
import yaml
import os

def load_config():
    '''
    This function is responsible for loading in the config file. 
    '''

    # Get the directory that this script is in
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Moving up one level, then into the config folder
    config_path = os.path.join(base_dir, '..', 'configs', 'pipeline_config.yaml')

    with open(config_path, 'r') as f:
        logging.info("Loading config file")
        return yaml.safe_load(f)

def fetch_data(config):
    '''
    This function is responsible for getting the data from NFLVerse.
    '''

    # Loading in config file and getting necessary seasons
    seasons = config['data']['seasons']

    # Loading in NFLVerse Data
    raw_data = nfl.load_pbp(seasons).to_pandas()
    logging.info("Loading NFL pbp data")

    return raw_data

def validate_data(data, config):
    '''
    This function is responsible for catching major errors in the data before saving it. 
    '''

    # Check for empty or none dataframe
    if data is None or data.empty:
        logging.info('Checking for none or empty dataframe')
        raise ValueError('Fetched NFLVerse data is empty or None')
    
    # Check for missing columns
    missing = set()

    # Checking for necessary columns to compute the target variable
    for col in ['play_type', 'run_location', 'pass_length']:
        if col not in data.columns:
            missing.add(col)

    # Checking features in config file
    for col in config['features']['raw_nflverse']:
        if col not in data.columns:
            missing.add(col)

    # Checking if necessary columns are present for historical feature calculation if enabled
    if config['features']['historical_features']['enabled']:
        for col in ['yards_gained', 'defteam', 'season', 'week']:
            if col not in data.columns:
                missing.add(col)

    if missing:
        logging.info("checking for missing columns")
        raise KeyError(f'Fetched data is missing columns: {missing}')
    
    return True

def save_raw_data(data, config):
    '''
    This function is responsible for saving the validated data to the designated folder for future use. 
    '''

    # Get necessary paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, '..', config['data']['paths']['raw_data_path'], config['data']['file_naming']['raw_data_name'])
    
    # Save raw Data
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    logging.info("Saving raw data")
    data.to_csv(save_path, index = False)
    
def main():
    '''
    This is the main logic of the program.
    '''
    try:
        # Getting config file
        config = load_config()

        # Getting data
        data = fetch_data(config)

        # If data is valid, save it to the desired raw data folder
        validate_data(data, config)
    
    except FileNotFoundError as e:
        logging.error(f"CONFIG ERROR: Could not find YAML file. Details: {e}")
        return 1
    
    except ValueError as e:
        logging.error(f"VALIDATION ERROR: The data failed integrity checks. Details: {e}")
        return 1
    
    except Exception as e:
        logging.error(f"UNEXPECTED ERROR: An unknown error occurred: {e}")
        return 1

    else:
        save_raw_data(data, config)
        return 0

if __name__ == "__main__":
    main()
