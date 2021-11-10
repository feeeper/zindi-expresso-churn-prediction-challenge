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


algorithms = {
    'cat_boost': CatBoostClassifier,
    'xgb': XGBClassifier,
    'lgb': LGBMClassifier,
    'sklgb': GradientBoostingClassifier
}


def train(df: pd.DataFrame, cfg: dict, path: str = None) -> Union[CatBoostClassifier, XGBClassifier, LGBMClassifier, GradientBoostingClassifier]:
    train_section: dict = cfg['train']
    train_algorithm_name: str = train_section['algorithm']
    train_algorithm_params: dict = train_section['params']
        
    log.info(f'Train {train_algorithm_name} with params:\n{json.dumps(train_section, indent=4)}')
    clf = algorithms[train_algorithm_name]

    target_column: str = train_section['target']
    skip_columns: list[str] = train_section['skip'] + [target_column]
    useful_features: list[str] = [x for x in list(df) if x not in skip_columns]

    final_test_predictions = []
    scores = []
    log.info(f'fold  score     elapsed (min)')
    total_sec: float = 0.0
    model_name: str = f'{dt.now()}'
    for fold in sorted(df['kfold'].unique()):
        start = dt.now()
        xtrain =  df[df.kfold != fold].reset_index(drop=True)
        xvalid = df[df.kfold == fold].reset_index(drop=True)
        
        ytrain = xtrain[target_column]
        yvalid = xvalid[target_column]
        
        xtrain = xtrain[useful_features]
        xvalid = xvalid[useful_features]

        model = clf(**train_algorithm_params)
        model.fit(xtrain, ytrain)

        preds_valid = model.predict_proba(xvalid)[:, 1]
        roc_auc = roc_auc_score(yvalid, preds_valid)
        sec_elapsed = (dt.now() - start).total_seconds()
        total_sec += sec_elapsed
        log.info(f'{fold:<6}{roc_auc:5f}  {sec_elapsed/60:5f}')
        scores.append(roc_auc)

        if path is not None:
            with open(f'{path}-fold-{fold}.pkl', 'xb') as model_file:
                pickle.to_pickle(model, model_file)

    log.info('-' * 30)
    log.info(f'Mean score: {np.mean(scores):5f}')
    log.info(f'Std: {np.std(scores):5f}')
    log.info(f'Total elapsed (min): {total_sec/60:5f}')

    return model


if __name__ == '__main__':
    default_config = './src/configs/base.yaml'
    default_model_name = f'./models/{dt.now().strftime("%Y-%m-%dT%H-%M")}'
    default_source_dataset = './data/raw/Train.zip'

    parser = ArgumentParser()    
    parser.add_argument('-src', help='path to source csv', type=str, default=default_source_dataset)
    parser.add_argument('-dst', help='path to result model', type=str, default=default_model_name)
    parser.add_argument('-cfg', help='path to config', default=default_config)
    parser.add_argument('-save', help='save models?', type=bool, default=True)
        
    args = parser.parse_args()
    log.info(args)
    
    if args.src.endswith('csv'):
        source_df = pd.read_csv(args.src)
    elif args.src.endswith('pkl'):
        source_df = pd.read_pickle(args.src)
    else:
        raise ValueError(f'Unkonwn file extension {args.src}')

    with open(args.cfg, mode='r', encoding='utf8') as f:
        cfg = yaml.safe_load(f)

    sample = source_df.sample(frac=cfg['train']['n_rows'], random_state=cfg['train']['random_state'])
    clf = train(sample, cfg, args.dst if args.save else None)
