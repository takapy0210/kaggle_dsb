import fire

from load_data import read_data_all
from create_feature import create_feature
from staging import staging_train, staging_test
from training import training


def main(dev=False) -> None:
    train, test, train_labels, submission = read_data_all(dev)
    features_train = create_feature(train)
    X_train, y_train = staging_train(train_labels, features_train)
    model = training(X_train, y_train)
    # feature_test = staging_test(test, create_feature(test))
    pass


if __name__ == '__main__':
    fire.Fire(main)
