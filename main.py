# -*- encoding: utf-8 -*-

import argparse
from utils.train import TongueClassifier


def parse_arguments():
    """
    Parse command-line arguments for training configuration.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a machine learning model.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to train (e.g., 'resnet50', 'vgg16')",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./croped_images",
        help="Path to the directory containing the training data",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=70,
        help="Number of epochs for training (default: 70)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate for the optimizer (default: 0.0001)",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=31,
        help="Random seeds for spliting datasets (default: 31)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    model_name = args.model
    data_dir = args.data_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    random_seed = args.random_seed

    cls = TongueClassifier(
        model_name, data_dir, batch_size, num_epochs, learning_rate, random_seed
    )

    print(f"Training {model_name} model with the following configuration:")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Random seed: {random_seed}")
    print(f"Device: {cls.device}\n")

    cls.run()
