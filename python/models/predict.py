'''
This script is responsible for allowing the user to make predictions with their previously trained model.

This script does not handle model evaluation. It serves as a resuable script to call trained models for prediction tasks.

Note: When calling this script the format is "python - m models.train_model --model [artifacts/trained_models/(specific model here)]"
'''

import argparse
import os
import json
import pickle
import pandas as pd
import logging
from models.train_model import load_model_config

def load_artifacts(path):
    '''
    This function is responsible for loading in the saved artifacts from a previously trained model.
    '''

    # Get necessary path.
    base_dir = os.path.dirname(os.path.abspath(__file__))

    artifacts_folder_path = os.path.join(base_dir, '..', path)

    # Retrieve model.pkl file.
    model_path = os.path.join(artifacts_folder_path, 'xgb_model.pkl')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Retrieve label_encoder.pkl file.
    encoder_path = os.path.join(artifacts_folder_path, 'label_encoder.pkl')

    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)

    # Retrieve feature_columns
    columns_path = os.path.join(artifacts_folder_path, 'feature_columns.json')
    with open(columns_path, 'r') as f:
        features = json.load(f)

    logging.info('Retrieving model, encoder and feature columns.')

    return model, encoder, features

def prepare_input(model_config, features):
    '''
    This function is responsible for taking in user prediction input and arranging it as an input dataframe for prediction.
    '''

    # Get user input
    prediction_input = model_config['prediction_input']

    # Convert to dataframe
    input_df = pd.DataFrame([prediction_input])

    # Validation
    missing = set(features) - set(input_df.columns)

    if missing:
        raise ValueError(f'Missing prediction features {missing}')
     
    input_df = input_df[features]

    logging.info('Converting user input into input dataframe for prediction.')

    return input_df

def predict_play(model, input_df, encoder):
    '''
    This function is responsible for predicting a play given the user input. 
    '''

    # Make prediction using model
    prediction = model.predict(input_df)
    decoded_prediction = encoder.inverse_transform(prediction)[0]

    probabilities = model.predict_proba(input_df)[0]
    confidence = max(probabilities)

    class_probabilities = dict(zip(encoder.classes_, probabilities))

    return decoded_prediction, confidence, class_probabilities

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        # Get Config
        model_config = load_model_config()

        # Initialize argument parser 
        parser = argparse.ArgumentParser(description = 'Run predictions using a trained model.')
        parser.add_argument(
            '--model',
            type = str,
            required = True,
            help = "Path to the trained model directory, starting with 'artifacts'."
        )

        args = parser.parse_args()

        model_path = args.model 
        
        model, encoder, features = load_artifacts(model_path)

        input_df = prepare_input(model_config, features)

        decoded_prediction, confidence, class_probabilities = predict_play(model, input_df, encoder)
    
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
        print('Predicted Play Type:', decoded_prediction)
        print('Confidence:', confidence, '\n')
        print('Class Probabilities', class_probabilities)

        return 0
    
if __name__ == "__main__":
    main()
    