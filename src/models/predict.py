import sys
import os

from icecream import ic
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from argparse import ArgumentParser
from typing import Union
import json
from datetime import datetime as dt

from pandas.io import pickle

from utils.colored_print import ColoredPrint

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import yaml

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier


log = ColoredPrint()


def predict(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    models: list[str] = config['predict']['models']
    predictions = []
    useful_columns: list[str] = [x for x in list(df) if x not in config['train']['skip']]

    for model_path in models:
        log.info(f'Predict with model "{model_path}"')
        with open(model_path, 'rb') as m:
            try:
                model = pickle.read_pickle(m)
            except Exception as e:
                log.err(f'Error while loading model {model_path}: {e}')

            preds = model.predict_proba(df[useful_columns])[:, 1]
            predictions.append(preds)

    log.info(np.array(predictions).shape)

    return pd.DataFrame({
        'user_id': df['user_id'],
        'CHURN': np.array(predictions).mean(axis=0)
    })


if __name__ == '__main__':
    default_config = './src/configs/base.yaml'
    default_submission_path = f'./data/submissions/{dt.now().strftime("%Y-%m-%dT%H-%M")}.csv'
    default_source_dataset = './data/raw/Test.zip'

    parser = ArgumentParser()    
    parser.add_argument('-src', help='path to source csv', type=str, default=default_source_dataset)
    parser.add_argument('-dst', help='path to submission', type=str, default=default_submission_path)
    parser.add_argument('-cfg', help='path to config', default=default_config)
        
    args = parser.parse_args()
    src = args.src
    acfg = args.cfg
    dst = args.dst
    log.info(args)
    
    # src = './data/interim/Test_fe1.pkl'
    # acfg = default_config
    # dst = default_submission_path

    if src.endswith('csv'):
        source_df = pd.read_csv(src)
    elif src.endswith('pkl'):
        source_df = pd.read_pickle(src)
    else:
        raise ValueError(f'Unkonwn file extension {src}')

    with open(acfg, mode='r', encoding='utf8') as f:
        cfg = yaml.safe_load(f)

    log.info(source_df.shape)

    result = predict(source_df, cfg)

    log.info(result.head(10))
    result.to_csv(dst, index=None)
    log.info(f'Submission saved to "{dst}"')
