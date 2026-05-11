'''
This script is responsible for finishing up the preprocessing step in this pipeline.

It will engineer historical features (if set to do so in config) and it will produce the final
preprocessed dataset (ready for training), saving the dataset to the specified location.
'''

import pandas as pd
import numpy as np
import os
import logging
from ingestion.load_data import load_pipeline_config

def get_cleaned_data(config):
    '''
    This function is responsible for retrieving the cleaned data from clean_data.py
    and returning it as a pandas dataframe.
    '''

    # Get location of data
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Get name of data
    cleaned_folder = config['data']['paths']['cleaned_data_path']
    df_name = config['data']['file_naming']['cleaned_data_name']

    # Moving up one level, then into the data folder
    data_path = os.path.join(base_dir, '..', cleaned_folder, df_name)

    logging.info(f'Loading cleaned data from {data_path}')

    return pd.read_csv(data_path, low_memory = False)

def create_historical_features(df, config):
    '''
    This function is responsible for engineering the historical features:

    Average Defensive pass rate: Average [(number of passes) / (total plays)] a defense faced over the specified time frame.

    Average yards per play allowed

    Average rush yards per play allowed
    '''

    if df is None:
        raise ValueError("DataFrame is None")
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    df = df.copy()

    # Retrieving necessary config variables to calculate historical features.
    rolling_window = config['features']['historical_features']['rolling_windows']

    # Engineering necessary features to calculate historical features. 
    df['rush_yards'] = df['yards_gained'].where(df['play_type'] == 'run')
    df['is_pass'] = (df['play_type'] == 'pass').astype(int)

    # Creating separate "summary" dataset to calculate historical features and eventually merge with the cleaned dataset.
    defsummary = (
        df.groupby(['defteam', 'season', 'week']).agg(
            avg_yds_allowed_per_play=('yards_gained', 'mean'),
            def_pass_rate=('is_pass', 'mean'),
            avg_rush_yds_allowed_play=('rush_yards', 'mean')
        )
        .reset_index()
    )
    logging.info('Creating Summary Dataset')

    defsummary = defsummary.sort_values(['defteam','season', 'week'])

    # Creating historical features all at once 
    grouped = defsummary.groupby(['defteam', 'season'])

    defsummary[['def_pass_rate_rolling', 'avg_yds_pp_alwd_rolling', 'avg_rush_yds_per_rush_alwd_rolling']] = (
        grouped[['def_pass_rate', 'avg_yds_allowed_per_play', 'avg_rush_yds_allowed_play']]
        .transform(lambda x: x.shift(1).rolling(rolling_window, min_periods = 1).mean())
    )

    # Merging historical features with existing dataframe.
    df = df.merge(defsummary[['defteam', 'season', 'week', 'def_pass_rate_rolling', 'avg_yds_pp_alwd_rolling', 'avg_rush_yds_per_rush_alwd_rolling']],
              on = ['defteam', 'season', 'week'],
              how = 'left')
    logging.info('Merging historical features with main dataframe')
    
    # Dropping null week one rolling values to preserve model simplicity.
    df = df.dropna(subset=['def_pass_rate_rolling', 'avg_yds_pp_alwd_rolling', 'avg_rush_yds_per_rush_alwd_rolling'])

    return df

def save_processed_data(df, config):
    '''
    This function is responsible for saving the processed dataset to its designated folder for future use. 
    '''

    # Get necessary paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, '..', config['data']['paths']['processed_data_path'], config['data']['file_naming']['processed_data_name'])

    # Save raw Data
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    logging.info("Saving processed data")
    df.to_csv(save_path, index = False)

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        config = load_pipeline_config()

        df = get_cleaned_data(config)

        if config['features']['historical_features']['enabled']:
            df = create_historical_features(df, config)

    except FileNotFoundError as e:
        logging.error(f"CONFIG ERROR: Could not find YAML file. Details: {e}")
        return 1
    
    except ValueError as e:
        logging.error(f"DATAFRAME ERROR: The data failed integrity checks. Details: {e}")
        return 1
    
    except Exception as e:
        logging.error(f"UNEXPECTED ERROR: An unknown error occurred: {e}")
        return 1
    
    else:
        save_processed_data(df, config)
        return 0

if __name__ == "__main__":
    main()
