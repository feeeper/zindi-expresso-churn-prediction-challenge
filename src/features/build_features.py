from enum import Enum
import sys
import os
import re
from typing import Any, Callable, Tuple
from pandas.core.frame import DataFrame

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

import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


log = ColoredPrint()


def log_call(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        log.info(f'{func.__name__} procesing: {args[0].name}')
        return func(*args, **kwargs)
    return wrapper


class Result(Enum):
    DataFrame = 0,
    Series = 1


@log_call
def one_hot_encode(train_series: pd.Series, test_series: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    # TODO Should one hot encode train and test datasets
    def __replace_non_letters_with_underscore(name: str) -> str:
        return re.sub('\W', '_', name).lower()

    ohe: OneHotEncoder = OneHotEncoder(handle_unknown='ignore', dtype=np.int0)
    sprs: csr_matrix = ohe.fit_transform(pd.DataFrame(train_series))
    fixed_series_name: str = __replace_non_letters_with_underscore(train_series.name)
    columns: list[str] = [f'{fixed_series_name}_{__replace_non_letters_with_underscore(col)}' for col in ohe.categories_[0]]
    train_tmp: pd.DataFrame = pd.DataFrame.sparse.from_spmatrix(sprs, columns=columns)

    sprs = ohe.transform(pd.DataFrame(test_series))
    test_tmp: pd.DataFrame = pd.DataFrame.sparse.from_spmatrix(sprs, columns=columns)
    return train_tmp, test_tmp


@log_call
def standard_scale(
    train_series: pd.Series,
    test_series: pd.Series,
    with_mean: bool=True,
    with_std: bool=True,
    target_series: pd.Series = None) -> Tuple[pd.Series, pd.Series]:

    standard_scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    
    transformed_train = standard_scaler.fit_transform(pd.DataFrame(train_series))
    transformed_test = standard_scaler.transform(pd.DataFrame(test_series))

    return pd.Series(pd.Series(transformed_train[:, 0], name=train_series.name)), pd.Series(pd.Series(transformed_test[:, 0], name=test_series.name))


@log_call
def min_max_scale(
    train_series: pd.Series,
    test_series: pd.Series,
    target_series: pd.Series = None) -> Tuple[pd.Series, pd.Series]:  
    scaler = MinMaxScaler()
    
    transformed_train = scaler.fit_transform(pd.DataFrame(train_series))
    transformed_test = scaler.transform(pd.DataFrame(test_series))

    return pd.Series(pd.Series(transformed_train[:, 0], name=train_series.name)), pd.Series(pd.Series(transformed_test[:, 0], name=test_series.name))


@log_call
def target_encoding(
    train_series: pd.Series,
    test_series: pd.Series,
    target_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    tmp: pd.DataFrame = pd.DataFrame({
        'cat_feature': train_series,
        'target': target_series
    })
    grouped: pd.DataFrame = tmp.groupby('cat_feature').agg(np.mean)
    grouped_map: dict = grouped.to_dict()['target']

    return train_series.map(grouped_map), test_series.map(grouped_map)


def null_action(any: Any) -> None:
    pass


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    result_train_df: pd.DataFrame = train_df.copy()
    result_test_df: pd.DataFrame = test_df.copy()

    feature_engineering_section = config['features']
    drop_columns: list[str] = []

    for feature_name, feature_settings in feature_engineering_section.items():
        methods: list[str] = feature_settings['methods']

        for method in methods:
            section: dict = feature_engineering[method]
            action: Callable = section['action']
            result: Result = section.get('result', Result.Series)

            # TODO: how can I handle drop columns?
            if action == null_action:
                drop_columns.append(feature_name)
                continue

            if result == Result.Series:
                result_train_df[f'{feature_name}_{method}'], result_test_df[f'{feature_name}_{method}'] = action(result_train_df[feature_name], result_test_df[feature_name], target_series=result_train_df['CHURN'])
            elif result == Result.DataFrame:
                train_tmp, test_tmp = action(result_train_df[feature_name], result_test_df[feature_name])
                
                result_train_df = pd.concat([result_train_df, train_tmp], axis='columns')
                result_test_df = pd.concat([result_test_df, test_tmp], axis='columns')
                
                del train_tmp
                del test_tmp

        drop_columns.append(feature_name)

    result_train_df = result_train_df.drop(drop_columns, errors='ignore', axis='columns')
    result_test_df = result_test_df.drop(drop_columns, errors='ignore', axis='columns')
    return result_train_df, result_test_df


feature_engineering = {
    'one_hot_encode': {
        'action': one_hot_encode,
        'result': Result.DataFrame
    },
    'standard_scale': {
        'action': standard_scale,
    },
    'min_max_scale': {
        'action': min_max_scale,
    },
    'target_encoding': {
        'action': target_encoding
    },
    'drop': {
        'action': null_action
    }
}


def read_data(path: str) -> pd.DataFrame:
    if path.endswith('csv'):
        df = pd.read_csv(path)
    elif path.endswith('pkl'):
        df = pd.read_pickle(path)
    else:
        raise ValueError(f'Unkonwn file extension {path}')
    return df

if __name__ == '__main__':
    default_config = './src/configs/base.yaml'

    parser = ArgumentParser()
    parser.add_argument('-train_src', help='path to train source csv', type=str, default='./data/raw/Train.zip')
    parser.add_argument('-train_dst', help='path to train result csv', type=str, default='./data/interim/Train_fe.csv')

    parser.add_argument('-test_src', help='path to test source csv', type=str, default='./data/raw/test.zip')
    parser.add_argument('-test_dst', help='path to test result csv', type=str, default='./data/interim/Test_fe.csv')

    parser.add_argument('-mode', help='how to save result data frame', choices=['pkl', 'csv'], default='csv')
    parser.add_argument('-cfg', help='path to config', default=default_config)
        
    args = parser.parse_args()
    log.info(args)
    
    source_train_df = read_data(args.train_src)
    source_test_df = read_data(args.test_src)

    with open(args.cfg, mode='r', encoding='utf8') as f:
        cfg = yaml.safe_load(f)
        
    train_dest_df, test_dest_df = build_features(source_train_df, source_test_df, cfg)

    if args.mode == 'csv':
        train_dest_df.to_csv(f'{args.train_dst}.csv', index=None)
        test_dest_df.to_csv(f'{args.test_dst}.csv', index=None)
    elif args.mode == 'pkl':
        train_dest_df.to_pickle(f'{args.train_dst}.pkl')
        test_dest_df.to_pickle(f'{args.test_dst}.pkl')
    else:
        raise ValueError(f'Unkonwn mode {args.mode}')
