
from load_data import read_data_all
from create_feature import create_feature


def main() -> None:
    train, test, train_labels, submission = read_data_all()
    feature_train = create_feature(train)
    feature_test = create_feature(test)
    pass


if __name__ == '__main__':
    main()
