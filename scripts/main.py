import os

import fire

from load_data import read_data_all
from create_feature import create_feature
from staging import staging_train, staging_test
from training import training

file_path = os.path.dirname(__file__)


def main(dev=False) -> str:
    train, test, train_labels, submission = read_data_all(dev)
    features_train = create_feature(train)
    X_train, y_train = staging_train(train_labels, features_train)
    model = training(X_train, y_train)
    features_test = create_feature(test)
    X_test = staging_test(test, features_test, submission)
    prediction = model.predict(X_test)  # TODO: 出力が適切な形になってない
    submission['accuracy_group'] = prediction
    submission.to_csv(os.path.join(file_path, '../data/output/submission.csv'), index=False)
    return 'Success!'


if __name__ == '__main__':
    fire.Fire(main)
