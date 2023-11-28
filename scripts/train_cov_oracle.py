#!/usr/bin/env python
# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Script communicating with the AFL++ custom mutator for Neuzz++ neural program smoothing.

- Train model for predicting edge or memory coverage based on seed content.
- Generate gradient information for each seed requested by AFL++ custom mutator.
- Communicate with custom mutator via named pipes.
"""
import argparse
import logging
import os
import pathlib
import sys
import time
from typing import Optional, Sequence

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from neuzzpp.data_loaders import CoverageSeedHandler, SeedFolderHandler, seed_data_generator
from neuzzpp.models import MLP, create_logits_model
from neuzzpp.mutations import compute_one_mutation_info
from neuzzpp.utils import (LRTensorBoard, create_work_folders,
                           model_needs_retraining)

# Configure logger - console
logger = logging.getLogger("neuzzpp")
logger.setLevel(logging.INFO)
console_logger = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_logger.setFormatter(log_formatter)
logger.addHandler(console_logger)


def train_model(args: argparse.Namespace, seed_handler: SeedFolderHandler):
    """
    Function loading the dataset from the seeds folder, building and training the model.

    Args:
        args: Input args of the script.
        seed_handler: Data loading object initialized for the seed folder that will be used
            for training.

    Returns:
        Trained model.
    """
    # (Re-)create data generators
    seed_handler.load_seeds_from_folder()
    seed_handler.split_dataset(random_seed=args.random_seed)
    training_generator = seed_handler.get_generator(batch_size=args.batch_size, subset="training")
    if args.val_split > 0.0:
        validation_generator = seed_handler.get_generator(
            batch_size=args.batch_size, subset="validation"
        )
        monitor_metric = "val_prc"
    else:
        validation_generator = None
        monitor_metric = "prc"

    # Compute class frequencies and weights
    _, initial_bias = seed_handler.get_class_weights()

    # Create training callbacks
    seeds_path = pathlib.Path(args.seeds)
    model_path = seeds_path.parent / "models"
    callbacks = []
    if not args.fast:
        model_save = ModelCheckpoint(
            str(model_path / "model.h5"),
            verbose=0,
            save_best_only=True,
            monitor=monitor_metric,
            mode="max",
        )
        callbacks.append(model_save)
    if args.early_stopping is not None:
        es = EarlyStopping(monitor=monitor_metric, patience=args.early_stopping, mode="max")
        callbacks.append(es)

    # Create model
    if not args.fast:
        tb_callback = LRTensorBoard(log_dir=str(model_path / "tensorboard"), write_graph=False)
        callbacks.append(tb_callback)
    model = MLP(
        input_dim=seed_handler.max_file_size,
        output_dim=seed_handler.max_bitmap_size,
        lr=args.lr,
        ff_dim=args.n_hidden_neurons,
        output_bias=initial_bias,
        fast=args.fast,
    )

    # Fit model
    model.model.fit(
        training_generator,
        steps_per_epoch=np.ceil(seed_handler.training_size / args.batch_size),
        validation_data=validation_generator,
        validation_steps=None
        if validation_generator is None
        else np.ceil(seed_handler.val_size / args.batch_size),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # Compute evaluation metrics on validation data
    if args.val_split > 0.0:
        class_threshold = 0.5  # Classification threshold, use default of 0.5
        val_gen = seed_data_generator(
            *seed_handler.val_set, args.batch_size, seed_handler.max_file_size
        )
        preds_val = model.model.predict(
            val_gen, steps=np.ceil(seed_handler.val_size / args.batch_size)
        )
        y_true = seed_handler.val_set[1]
        y_pred = preds_val > class_threshold
        acc = accuracy_score(y_true.flatten(), y_pred.flatten())
        tp = np.sum(np.where(np.logical_and(y_true == y_pred, y_pred == 1), True, False), axis=0)
        tn = np.sum(np.where(np.logical_and(y_true == y_pred, y_pred == 0), True, False), axis=0)
        fp = np.sum(np.where(np.logical_and(y_true != y_pred, y_pred == 1), True, False), axis=0)
        fn = np.sum(np.where(np.logical_and(y_true != y_pred, y_pred == 0), True, False), axis=0)
        assert (tp + tn + fp + fn == seed_handler.val_size).all()
        assert tp.shape[0] == tn.shape[0] == fp.shape[0] == fn.shape[0] == y_true.shape[1]
        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fn) != 0)
        f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp + fn) != 0)
        prc = tf.keras.metrics.AUC(curve="PR", multi_label=True, num_labels=y_true.shape[1])
        prc.update_state(y_true, y_pred)
        print(
            f"Acc: {acc}, prec: {prec.mean()}, recall: {recall.mean()}, f1: {f1.mean()}, pr-auc: {prc.result().numpy()}"
        )

    return model.model


def create_parser() -> argparse.ArgumentParser:
    """
    Create and return the parser instance for the CLI arguments.

    Returns:
        Parser object.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-l", "--max_len", help="maximum seed length", type=int, default=None)
    parser.add_argument(
        "-p",
        "--percentile_len",
        help="percentile of seed length to keep as maximum seed length (1-100); "
        "ignored if max_len is provided",
        type=int,
        default=80,
    )
    parser.add_argument(
        "-e", "--epochs", help="number of epochs for model training", type=int, default=100
    )
    parser.add_argument(
        "-b", "--batch_size", help="batch size for model training", type=int, default=32
    )
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
    parser.add_argument("-s", "--early_stopping", help="early stopping patience", type=int)
    parser.add_argument(
        "--n_hidden_neurons", help="number of neurons in hidden layer", type=int, default=4096
    )
    parser.add_argument(
        "-v",
        "--val_split",
        help="amount of data between 0 and 1 to reserve for validation",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-r", "--random_seed", help="seed for random number generator", type=int, default=None
    )
    parser.add_argument(
        "-f",
        "--fast",
        help="train faster by skipping detailed model evaluation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-c",
        "--cov",
        help="type of coverage to measure",
        type=str,
        choices=["edge"],  # Only edge coverage supported for now
        default="edge",
    )
    parser.add_argument("input_pipe", help="", type=str)
    parser.add_argument("output_pipe", help="", type=str)
    parser.add_argument(
        "seeds", help="path to seeds folder, usually named `queue` for AFL++", type=str
    )
    parser.add_argument(
        "target",
        help="target program and arguments",
        type=str,
        nargs=argparse.REMAINDER,
        metavar="target [target_args]",
    )
    return parser


def main(argv: Sequence[str] = tuple(sys.argv)) -> None:
    n_seeds_last_training: int = 0
    time_last_training: int = 0

    parser = create_parser()
    args = parser.parse_args(argv[1:])

    # Configure logger - file
    seeds_path = pathlib.Path(args.seeds)
    file_logger = logging.FileHandler(seeds_path.parent / "training.log")
    file_logger.setFormatter(log_formatter)
    logger.addHandler(file_logger)

    # Check that input and output named pipes exist
    input_pipe = pathlib.Path(args.input_pipe)
    output_pipe = pathlib.Path(args.output_pipe)
    if not input_pipe.is_fifo() or not output_pipe.is_fifo():
        raise ValueError("Input or output pipes do not exist.")

    # Validate inputs
    if args.percentile_len <= 0 or args.percentile_len > 100:
        raise ValueError(
            f"Invalid `percentile_len`. "
            f"Expected integer in [1, 100], received: {args.percentile_len}."
        )
    if args.val_split < 0.0 or args.val_split >= 1.0:
        raise ValueError(
            f"Invalid `val_split`. Expected value in [0, 1), received: {args.val_split}."
        )
    create_work_folders(seeds_path.parent)

    data_loader = CoverageSeedHandler(  # Only edge coverage supported for now
        seeds_path,
        args.target,
        args.max_len,
        args.percentile_len,
        args.val_split,
    )
    model: Optional[MLP] = None
    out_pipe = open(output_pipe, "w")
    max_grads = os.environ.get("NEUZZPP_MAX_GRADS")
    n_grads = None if max_grads is None else int(max_grads)
    with open(input_pipe, "r") as seed_fifo:
        for seed_name in seed_fifo:
            # (Re-)train model if necessary
            if (
                model_needs_retraining(seeds_path, time_last_training, n_seeds_last_training)
                or model is None
            ):
                # Update info for model retraining
                n_seeds_last_training = len(list(seeds_path.glob("id*")))
                time_last_training = int(time.time())

                model = train_model(args, data_loader)
                grad_model = create_logits_model(model)

            # Generate gradients for requested seed
            target_path = pathlib.Path(str(seeds_path) + seed_name.strip())
            sorting_index_lst, gradient_lst = compute_one_mutation_info(
                grad_model, target_path, n_grads
            )
            out_pipe.write(",".join(sorting_index_lst) + "|" + ",".join(gradient_lst) + "\n")
            out_pipe.flush()
    out_pipe.close()


if __name__ == "__main__":
    main()
