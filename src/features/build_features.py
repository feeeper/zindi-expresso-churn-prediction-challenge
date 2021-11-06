from enum import Enum
import sys
import os
import re
from typing import Any, Callable

from tqdm import tqdm
import yaml
from icecream import ic

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from argparse import ArgumentParser
from configparser import ConfigParser

import pandas as pd
import numpy as np
from scipy.sparse.csr import csr_matrix

from utils.colored_print import ColoredPrint

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures


log = ColoredPrint()


class Result(Enum):
    DataFrame = 0,
    Series = 1


def one_hot_encode(series: pd.Series) -> pd.DataFrame:
    def __fix_series_name(name: str) -> str:
        return re.sub('\W', '_', name).lower()

    log.info(f'one_hot_encode processing: {series.name}')
    ohe: OneHotEncoder = OneHotEncoder(handle_unknown='ignore', dtype=np.int0)
    sprs: csr_matrix = ohe.fit_transform(pd.DataFrame(series))
    fixed_series_name: str = __fix_series_name(series.name)
    tmp: pd.DataFrame = pd.DataFrame.sparse.from_spmatrix(sprs, columns=[f'{fixed_series_name}_{__fix_series_name(col)}' for col in ohe.categories_[0]])
    return tmp


def standard_scale(series: pd.Series, with_mean: bool=True, with_std: bool=True) -> pd.Series:
    log.info(f'standard_scale processing: {series.name}')
    standard_scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    tmp = pd.DataFrame(series)
    transformed = standard_scaler.fit_transform(tmp)
    return pd.Series(pd.Series(transformed[:, 0], name=series.name))


def null_action(any: Any) -> None:
    pass


def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    result_df: pd.DataFrame = df.copy()
    feature_engineering_section = config['features']
    drop_columns: list[str] = []
    
    for feature_name, feature_settings in tqdm(feature_engineering_section.items()):
        method: str = feature_settings['method']
        section: dict = feature_engineering[method]
        action: Callable = section['action']
        result: Result = section.get('result', Result.Series)

        # todo: how can I handle drop columns?
        if action == null_action:
            drop_columns.append(feature_name)
            continue

        if result == Result.Series:
            result_df[feature_name] = action(result_df[feature_name])
        elif result == Result.DataFrame:
            ic(result_df[result_df[feature_name].isna()])
            tmp: pd.DataFrame = action(result_df[feature_name])
            result_df = result_df.drop(feature_name, axis='columns')
            result_df = pd.concat([result_df, tmp], axis='columns')
            del tmp

    ic(drop_columns)
    result_df = result_df.drop(drop_columns, errors='ignore', axis='columns')
    return result_df


feature_engineering = {
    'one_hot_encode': {
        'action': one_hot_encode,
        'result': Result.DataFrame
    },
    'standard_scale': {
        'action': standard_scale,
    },
    'drop': {
        'action': null_action
    }
}


if __name__ == '__main__':
    default_config = './src/configs/base.yaml'

    parser = ArgumentParser()
    parser.add_argument('-src', help='path to source csv', type=str, default='./data/raw/Train.zip')
    parser.add_argument('-dst', help='path to result csv', type=str, default='./data/interim/Train.csv')
    parser.add_argument('-mode', help='how to save result data frame', choices=['pkl', 'csv'], default='csv')
    parser.add_argument('-cfg', help='path to config', default=default_config)
        
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
    dest_df = build_features(source_df, cfg)

    if args.mode == 'csv':
        dest_df.to_csv(f'{args.dst}.csv', index=None)
    elif args.mode == 'pkl':
        dest_df.to_pickle(f'{args.dst}.pkl')
    else:
        raise ValueError(f'Unkonwn mode {args.mode}')
