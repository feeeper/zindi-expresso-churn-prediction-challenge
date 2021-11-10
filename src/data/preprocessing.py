import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from argparse import ArgumentParser
import yaml
from icecream import ic

from typing import Callable, Union
from enum import Enum

from sklearn.model_selection import KFold
import pandas as pd
from tqdm import tqdm
pd.options.mode.chained_assignment = None

from utils.colored_print import ColoredPrint


log = ColoredPrint()


class CategoricalFeatureFillNanMode(Enum):
    MOST_COMMON = 1
    NEW_VALUE = 2


class NumericalFeatureFillNanMode(Enum):
    MOST_COMMON = 1
    NEW_VALUE = 2
    MEAN = 3
    MEDIAN = 4


class FeatureType(Enum):
    Categorical = 1
    Numerical = 2


nan_values_per_column = {
    'ZONE2': 93.64805241108833,
    'ZONE1': 92.1208348189084,
    'TIGO': 59.88798764001545,
    'DATA_VOLUME': 49.22977575244377,
    'TOP_PACK': 41.90222316308643,
    'FREQ_TOP_PACK': 41.90222316308643,
    'ORANGE': 41.561190836973,
    'REGION': 39.42804431470422,
    'ON_NET': 36.52077391033069,
    'MONTANT': 35.13101843598657,
    'FREQUENCE_RECH': 35.13101843598657,
    'REVENUE': 33.70621267492647,
    'ARPU_SEGMENT': 33.70621267492647,
    'FREQUENCE': 33.70621267492647,
    'user_id': 0.0,
    'TENURE': 0.0,
    'MRG': 0.0,
    'REGULARITY': 0.0,
    'CHURN': 0.0
}

unique_values_per_column = {
    'user_id': 2154048,
    'DATA_VOLUME': 41550,
    'REVENUE': 38114,
    'ARPU_SEGMENT': 16535,
    'ON_NET': 9884,
    'MONTANT': 6540,
    'ORANGE': 3167,
    'TIGO': 1315,
    'ZONE1': 612,
    'ZONE2': 486,
    'FREQ_TOP_PACK': 245,
    'TOP_PACK': 140,
    'FREQUENCE_RECH': 123,
    'FREQUENCE': 91,
    'REGULARITY': 62,
    'REGION': 14,
    'TENURE': 8,
    'CHURN': 2,
    'MRG': 1
}


def remove_nan_columns(df: pd.DataFrame, threshold: float = 90.0) -> pd.DataFrame:
    columns: list[str] = [x[0] for x in nan_values_per_column.items() if x[1] < threshold]
    result_columns = [x for x in columns if x in list(df)]
    return df[result_columns]


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns: list[str] = [x[0] for x in unique_values_per_column.items() if x[1] == 1]
    result_columns = [x for x in columns if x in list(df)]
    return df[result_columns]


def fill_nan_categorical_feature(
        train_series: pd.Series,
        mode: CategoricalFeatureFillNanMode = CategoricalFeatureFillNanMode.NEW_VALUE,
        new_value: Union[str, float] = 'OTHER') -> pd.Series:
    if mode == CategoricalFeatureFillNanMode.NEW_VALUE:
        log.info(f'Column "{train_series.name}" fill None with "{new_value}" ({mode})')
        return train_series.fillna(new_value)
    if mode == CategoricalFeatureFillNanMode.MOST_COMMON:
        most_commons: pd.Series = train_series.mode(dropna=True)

        if most_commons.shape[0] > 1:
            log.warn(f'There are more than one most common values. '
                  f'Values [{", ".join(most_commons)}] were met {train_series.value_counts()[most_commons[0]]} times. '
                  f'The first ("{most_commons[0]}") was selected. '
                  f'You can use CategoricalFeatureFillNanMode.NEW_VALUE for set exact value for `None` values.')
        most_common = most_commons[0]
        log.info(f'Column "{train_series.name}" fill None with "{most_common}" ({mode})')
        return train_series.fillna(most_common)
    raise ValueError(f'Unsupported mode was passed: {mode}')


def fill_nan_numerical_features(
        series: pd.Series,
        mode: NumericalFeatureFillNanMode = NumericalFeatureFillNanMode.NEW_VALUE,
        new_value: float = -1) -> pd.Series:
    if mode == NumericalFeatureFillNanMode.NEW_VALUE:
        return fill_nan_categorical_feature(series, CategoricalFeatureFillNanMode.NEW_VALUE, new_value)
    if mode == NumericalFeatureFillNanMode.MOST_COMMON:
        return fill_nan_categorical_feature(series, CategoricalFeatureFillNanMode.MOST_COMMON)
    if mode == NumericalFeatureFillNanMode.MEAN:
        mean = series.mean()
        log.info(f'Column "{series.name}" fill None with "{mean}" ({mode})')
        return series.fillna(mean)
    if mode == NumericalFeatureFillNanMode.MEDIAN:
        median = series.median()
        log.info(f'Column "{series.name}" fill None with "{median}" ({mode})')
        return series.fillna(median)


def create_folds(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    n_splits: int = int(cfg['general']['kfolds'])
    df["kfold"] = -1
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_indicies, valid_indicies) in enumerate(kf.split(X=df)):
        df.loc[valid_indicies, "kfold"] = fold
    return df


def prepare(
    train_df: pd.DataFrame,
    config: dict) -> pd.DataFrame:
    try:
        log.info('Clean dataset: remove constant columns')
        train_result_df: pd.DataFrame = remove_constant_columns(train_df)

        log.info(f'Clean dataset: remove columns with None more than {config["general"]["none_threshold"]}')
        threshold: float = float(config["general"]["none_threshold"]) * 100
        train_result_df = remove_nan_columns(train_result_df, threshold)

        features = config['features']
        drop_columns: list[str] = []
        for feature_name, feature_settings in features.items():
            if feature_name not in train_df:
                log.warn(f'Column "{feature_name}" not presented in the data frame')
                continue

            if feature_settings['method'] == 'drop':
                drop_columns.append(feature_name)
                continue

            try:
                if FeatureType[feature_settings['type']] == FeatureType.Categorical:
                    train_result_df[feature_name] = fill_nan_categorical_feature(train_df[feature_name], CategoricalFeatureFillNanMode[feature_settings['fillna']])
                else:
                    train_result_df[feature_name] = fill_nan_numerical_features(train_df[feature_name], NumericalFeatureFillNanMode[feature_settings['fillna']])
            except KeyError as ke:
                log.warn(f'Feature {feature_name} has incorrect type ({feature_settings["type"]})')
                ic(feature_settings['type'])
                ic(type(feature_settings['type']))

        train_result_df = train_result_df.drop(drop_columns, errors='ignore', axis='columns')
        train_result_df = create_folds(train_result_df, config)
    except Exception as e:
        log.err(e)
        raise

    return train_result_df



if __name__ == '__main__':
    default_config = './src/configs/baseline.ini'

    parser = ArgumentParser()
    parser.add_argument('-src', help='path to source csv', type=str, default='./data/raw/Train.zip')
    parser.add_argument('-dst', help='path to result csv', type=str, default='./data/interim/Train')
    parser.add_argument('-mode', help='how to save result data frame', choices=['pkl', 'csv'], default='csv')
    parser.add_argument('-cfg', help='path to config', default=default_config)
        
    args = parser.parse_args()
    log.info(args)

    with open(args.cfg, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # ic(FeatureType[config['features']['REGION']['type']])
    # raise

    train_source_df = pd.read_csv(args.src)
    train_dest_df = prepare(train_source_df, config)

    mode: str = args.mode
    if mode == 'csv':
        train_dest_df.to_csv(f'{args.dst}.csv', index=None)
    elif mode == 'pkl':
        train_dest_df.to_pickle(f'{args.dst}.pkl')
    else:
        raise ValueError(f'Unkonwn mode {mode}')
else:
    print(__name__)
