import pandas as pd
import yaml
import os

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data(path):
    """Charge les données brutes depuis le chemin spécifié"""
    return pd.read_csv(path)

