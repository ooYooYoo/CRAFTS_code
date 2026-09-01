import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if not isinstance(v, str):
        raise argparse.ArgumentTypeError("Boolean value expected.")

    value = v.lower()
    if value == "true":
        return True
    if value == "false":
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def recommend_parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Dataset
    parser.add_argument(
        "--dataset",
        nargs="?",
        default="Instruments",
        choices=["Instruments"],
        help="main-experiment dataset",
    )
    parser.add_argument(
        "--data_path",
        nargs="?",
        default="data/clean/",
        help="dataset root directory",
    )
    parser.add_argument(
        "--training_data",
        nargs="?",
        default="/train.txt",
        help="training split file",
    )
    parser.add_argument(
        "--val_data",
        nargs="?",
        default="/val.txt",
        help="validation split file",
    )
    parser.add_argument(
        "--test_data",
        nargs="?",
        default="/test.txt",
        help="test split file",
    )
    parser.add_argument(
        "--meta_file",
        nargs="?",
        default="/id2meta.json",
        help="item metadata file",
    )
    parser.add_argument(
        "--load_meta",
        type=str2bool,
        default=True,
        help="whether to load item metadata",
    )
    parser.add_argument(
        "--split_data",
        type=str2bool,
        default=True,
        help="whether to prepare train/validation/test benchmark files",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default="loo",
        help="data split method: loo, ratio, or temporal",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="test-set ratio for ratio/temporal splitting",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="validation-set ratio for ratio/temporal splitting",
    )

    # Recommender
    parser.add_argument(
        "--model_type",
        type=str,
        default="implicit",
        help="recommender model type",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ARS",
        help="recommender model name",
    )
    parser.add_argument(
        "--maxEpoch",
        type=int,
        default=4,
        help="number of recommender training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="recommender training batch size",
    )
    parser.add_argument(
        "--emb_size",
        type=int,
        default=64,
        help="embedding size",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="number of model layers",
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=1e-4,
        help="regularization weight",
    )
    parser.add_argument(
        "--lRate",
        type=float,
        default=0.005,
        help="learning rate",
    )
    parser.add_argument(
        "--dropout",
        type=str2bool,
        default=True,
        help="whether to use dropout",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.3,
        help="dropout rate",
    )
    parser.add_argument(
        "--cuda",
        type=str2bool,
        default=True,
        help="whether to use CUDA",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="GPU id",
    )
    parser.add_argument(
        "--seed",
        nargs="?",
        type=int,
        default=2018,
        help="random seed",
    )
    parser.add_argument(
        "--topK",
        nargs="?",
        default="1",
        help="evaluation cutoff",
    )

    # Save / load
    parser.add_argument(
        "--load",
        type=str2bool,
        default=True,
        help="load an existing recommender model",
    )
    parser.add_argument(
        "--save",
        type=str2bool,
        default=True,
        help="save recommender artifacts",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./artifacts/models/",
        help="output directory for recommender artifacts",
    )

    return parser.parse_known_args()[0]
