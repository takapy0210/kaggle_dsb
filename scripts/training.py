import pandas as pd
import lightgbm as lgb

params = {
    'n_estimators': 2000,
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'subsample': 0.75,
    'subsample_freq': 1,
    'learning_rate': 0.04,
    'feature_fraction': 0.9,
    'max_depth': 15,
    'lambda_l1': 1,
    'lambda_l2': 1,
    'verbose': 100,
    # 'early_stopping_rounds': 100,
    'eval_metric': 'cappa'
}


def training(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model
