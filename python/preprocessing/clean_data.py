'''
This program is responsible for taking the raw data from load_data.py and doing initial cleaning
and target feature engineering. It will save the cleaned dataset to the folder designated in 
the pipeline_config.yaml file. Further feature engineering will take place in build_features.py.
'''

import pandas as pd
import numpy as np
import os
import logging
from ingestion.load_data import load_pipeline_config

def get_raw_data(config):
    '''
    This function is responsible for retrieving the raw data from load_data.py
    and returning it as a pandas dataframe.
    '''

    # Get location of data
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Get name of data
    raw_folder = config['data']['paths']['raw_data_path']
    df_name = config['data']['file_naming']['raw_data_name']

    # Moving up one level, then into the data folder
    data_path = os.path.join(base_dir, '..', raw_folder, df_name)

    logging.info(f'Loading raw data from {data_path}')

    return pd.read_csv(data_path, low_memory = False)

def initial_clean(df):
    '''
    This function is responsible for initally cleaning the data.
    '''

    # Filter out rows that are outside the scope of the model
    df = df[
        (df['play_type'].isin(['run', 'pass'])) & 
        (df['special'] == 0) & 
        (df['two_point_attempt'] == 0) & 
        (df['sack'] == 0) & 
        (df['qb_scramble'] == 0) & 
        (df['qb_kneel'] == 0) & 
        (df['qb_spike'] == 0)
        ].copy()
    logging.info('Filtering out rows that are outside the scope of the model.')
    
    # Filter out invalid rows where a run play has no direction or a pass play has no length indicator
    run_mask = (df['play_type'] == 'run') & (df['run_location'].isna())
    pass_mask = (df['play_type'] == 'pass') & (df['pass_length'].isna())

    df = df[~(run_mask | pass_mask)]

    logging.info('Returning initially cleaned df.')
    return df

def classify_play(df):
    '''
    This function is specifically responsible for building the target variable "play_subtype".
    '''
    
    # Conditions for each play subtype
    conditions = [
        (df['play_type'] == 'run') & (df['run_location'] == 'middle'),
        (df['play_type'] == 'run') & (df['run_location'].isin(['left', 'right'])),
        (df['play_type'] == 'pass') & (df['pass_length'] == 'short'),
        (df['play_type'] == 'pass') & (df['pass_length'] == 'deep')
    ]

    # Classifying rows based on what data they have
    choices = ['inside run', 'outside run', 'short pass', 'deep pass']

    df = df.copy()

    df['play_subtype'] = np.select(conditions, choices, default='other')
    logging.info(f"Adding play subtype target column. Subtype distribution: {df['play_subtype'].value_counts().to_dict()}")
    return df

def filter_rows(df):
    '''
    This function is responsible for a final validation of the cleaned data. It will remove any rows where 
    play_subtype = 'other' and ensure that the dataframe is not empty after filters have been applied.
    '''

    # Ensuring validity of df.
    if df is None or df.empty:
        logging.info('Checking for none or empty dataframe')
        raise ValueError('Filtered dataframe is empty or None')

    # Dropping rows with invalid target column.
    initial_count = len(df)
    df = df[df['play_subtype'] != 'other'].copy()
    final_count = len(df)

    # Ensuring validity of df.
    if df is None or df.empty:
        logging.info('Checking for none or empty dataframe')
        raise ValueError('Filtered dataframe is empty or None')
    
    logging.info(f'Filtering complete. dropped {initial_count - final_count} unclassified rows.')
    return df

def save_cleaned_data(df, config):
    '''
    This function is responsible for saving the cleaned dataset to its designated folder for future use. 
    '''

    # Get necessary paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, '..', config['data']['paths']['cleaned_data_path'], config['data']['file_naming']['cleaned_data_name'])

    # Save raw Data
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    logging.info("Saving cleaned data")
    df.to_csv(save_path, index = False)

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        config = load_pipeline_config()

        df = get_raw_data(config)

        df_filtered = initial_clean(df)

        df_labeled = classify_play(df_filtered)

        df_cleaned = filter_rows(df_labeled)
    
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
        save_cleaned_data(df_cleaned, config)
        return 0

if __name__ == "__main__":
    main()
