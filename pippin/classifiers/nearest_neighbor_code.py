import argparse
import numpy as np
import pandas as pd
import logging
import sys
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def setup_logging():
    fmt = "[%(levelname)8s |%(funcName)21s:%(lineno)3d]   %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        handlers=[handler, logging.FileHandler("nn.log")],
    )
    logging.getLogger("matplotlib").setLevel(logging.ERROR)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fitres_file",
        help="the name of the fitrres file to load. For example: somepath/FITOPT000.FITRES",
    )
    parser.add_argument(
        "-p", "--predict", help="If in predict mode", action="store_true"
    )
    parser.add_argument(
        "-m", "--model", help="Pickled model to load", default="model.pkl", type=str
    )
    parser.add_argument(
        "-d",
        "--done_file",
        help="Location to write done file",
        default="done.txt",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--features",
        help="Space separated string of features out of fitres",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--types",
        help="Ia types, space separated list",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV of predictions",
        type=str,
        default="predictions.csv",
    )
    parser.add_argument(
        "-n", "--name", help="Column name for probability", type=str, default="PROB"
    )
    args = parser.parse_args()
    return args


def sanitise_args(args):
    """Set up defaults and do some sanity checks"""

    if args.features is None:
        args.features = [
            "zHD",
            "x1",
            "c",
            "cERR",
            "x1ERR",
            "mBERR",
            "COV_x1_c",
            "COV_x1_x0",
            "COV_c_x0",
            "FITPROB",
        ]

    if args.types is None:
        args.types = [1, 101]

    logging.info(f"Input fitres_file is {args.fitres_file}")
    assert os.path.exists(args.fitres_file), f"File {args.fitres_file} does not exist"

    assert (
        " " not in args.name
    ), f"Prob column name '{args.name}' should not have spaces"
    return args


def get_features(filename, features, types):
    df = pd.read_csv(filename, delim_whitespace=True, comment="#")
    for f in features:
        assert (
            f in df.columns
        ), f"Features {f} is not in DataFrame columns {list(df.columns)}"
    assert "TYPE" in df.columns, f"DataFrame does not have a TYPE column!"

    X = df[features].values
    y = np.isin(df["TYPE"].values, types)

    return df["CID"], X, y


def train(args):
    args = sanitise_args(args)
    logging.info(f"Training model on file {args.fitres_file}")

    _, X, y = get_features(args.fitres_file, args.features, args.types)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.05, random_state=0
    )

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=50, algorithm="kd_tree")),
        ]
    )

    logging.info(f"Training NN on feature matrix {X.shape}")
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    logging.info(f"Evaluating on 5% test set: got accuracy {score:0.4f}")

    with open(args.model, "wb") as f:
        pickle.dump(clf, f)
    logging.info(f"Saved trained model out to {args.model}")


def predict(args):
    args = sanitise_args(args)
    logging.info(
        f"Predicting model on file {args.fitres_file} using pickle {args.model}"
    )
    assert os.path.exists(args.model), f"Pickle {args.model} does not exist!"
    with open(args.model, "rb") as f:
        clf = pickle.load(f)

    cids, X, _ = get_features(args.fitres_file, args.features, args.types)
    y = clf.predict_proba(X)

    df_output = pd.DataFrame({"SNID": cids, args.name: y[:, 1]})
    df_output.to_csv(args.output, float_format="%0.2f", index=False)
    logging.info(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    args = None
    try:
        args = get_args()
        setup_logging()

        if args.predict:  # Run in training mode
            predict(args)
        else:
            train(args)

        with open(args.done_file, "w") as f:
            f.write("SUCCESS")
    except Exception as e:
        logging.exception(e, exc_info=True)
        with open(args.done_file, "w") as f:
            f.write("FAILURE")
