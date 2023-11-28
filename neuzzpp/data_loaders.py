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
"""Module containing data loaders for model training and evaluation."""
import logging
import os
import pathlib
from typing import Callable, Dict, Generator, List, Optional, Set, Tuple, Union

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from neuzzpp.preprocess import create_bitmap_from_raw_coverage, create_path_coverage_bitmap
from neuzzpp.utils import get_max_file_size

logger = logging.getLogger(__name__)


class SeedFolderHandler:
    def __init__(
        self,
        seeds_path: pathlib.Path,
        target: List[str],
        bitmap_func: Callable[
            [List[str], List[pathlib.Path]], Dict[pathlib.Path, Optional[Set[int]]]
        ],
        max_len: Optional[int] = None,
        percentile_len: Optional[int] = None,
        validation_split: float = 0.0,
    ) -> None:
        """
        Initialize a dataset handler based on a seed folder.

        Args:
            seeds_path: Path to the seeds folder.
            target: Name of the fuzzing target in a callable format with its arguments.
            max_len: Optional limit for the maximum seed length.
            percentile_len: Optional length percentile to keep as maximum seed length (1-100);
                ignored if `max_len` is provided.
            validation_split: Percentage of data to reserve for model validation.
        """
        if validation_split and not 0 < validation_split < 1:
            err = f"`validation_split` must be between 0 and 1, received: {validation_split}"
            logger.exception(err)
            raise ValueError(err)
        if max_len is not None and max_len <= 0:
            err = f"Maximum seed length must be greater than 0, received: {max_len}"
            logger.exception(err)
            raise ValueError(err)

        self.validation_split = validation_split
        self.max_len = max_len
        self.percentile_len = percentile_len
        self.seeds_path = seeds_path
        self.target = target
        self.bitmap_func = bitmap_func
        self.seed_list: Optional[List[pathlib.Path]] = None
        self.raw_coverage_info: Dict[pathlib.Path, Set[int]] = {}
        self.reduced_bitmap: Optional[np.ndarray] = None

    def load_seeds_from_folder(self) -> None:
        """
        Reload seeds information from queue folder. Only new seeds are considered.
        The coverage bitmap is recomputed and stored, along with other dataset info.
        """
        # Determine if there are any new seeds in folder since last loaded
        new_seed_list: List[pathlib.Path] = [
            seed for seed in self.seeds_path.glob("*") if seed.is_file()
        ]
        if self.seed_list is not None:
            new_seeds = list(set(new_seed_list) - set(self.seed_list))
        else:
            new_seeds = new_seed_list

        # Get coverage for new seeds and recompute coverage bitmap
        if new_seeds:
            self.raw_coverage_info.update(self.bitmap_func(self.target, new_seeds))
            self.seed_list, self.reduced_bitmap = create_bitmap_from_raw_coverage(
                self.raw_coverage_info
            )

        # Compute "optimal" max seed length if not provided - this dictates the model input size
        max_file_size = get_max_file_size(self.seeds_path)
        if self.max_len is None:
            max_len = compute_input_size_from_seeds(self.seeds_path, percentile=self.percentile_len)
            logger.info(
                f"No max length provided. "
                f"Using {max_len} based on existing seeds ({self.percentile_len}%)."
            )
        else:
            max_len = self.max_len
        self.max_file_size = min(max_file_size, max_len)
        self.max_bitmap_size = self.reduced_bitmap.shape[1]

    def split_dataset(self, random_seed: Optional[int] = None) -> None:
        """
        Split the seeds and corresponding bitmap into training and validation sets.
        The obtained split is assigned as attributes of the current object.

        Args:
            random_seed: Seed for random number generator impacting the seed split between
                training and validation.
        """
        if self.validation_split > 0.0:
            train_list, val_list, train_bitmaps, val_bitmaps = train_test_split(
                self.seed_list,
                self.reduced_bitmap,
                test_size=self.validation_split,
                random_state=random_seed,
            )
            train_list = [seed.as_posix() for seed in train_list]
            val_list = [seed.as_posix() for seed in val_list]
            self.training_set = (train_list, train_bitmaps)
            self.training_size = len(train_list)
            self.val_set = (val_list, val_bitmaps)
            self.val_size = len(val_list)
        else:
            train_list = [seed.as_posix() for seed in self.seed_list]
            self.training_set = (train_list, self.reduced_bitmap)
            self.training_size = len(train_list)

    def get_generator(self, batch_size: int, subset: str):
        """
        Create data generator based on the content of the seed folder.

        Args:
            batch_size: Batch size that the generator should produce.
            subset: "training" or "validation". "validation" produces an error if no data was
                set aside for validation.

        Returns:
            Data generator for training or validation as used by `tf.keras.model.fit`.
        """
        if subset == "training":
            return tf.data.Dataset.from_generator(
                seed_data_generator,
                output_signature=(
                    tf.TensorSpec(shape=(None, self.max_file_size), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, self.max_bitmap_size), dtype=tf.int32),
                ),
                args=[*self.training_set, batch_size, self.max_file_size],
            )
        elif subset == "validation":
            if self.validation_split > 0.0:
                return tf.data.Dataset.from_generator(
                    seed_data_generator,
                    output_signature=(
                        tf.TensorSpec(shape=(None, self.max_file_size), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, self.max_bitmap_size), dtype=tf.int32),
                    ),
                    args=[*self.val_set, batch_size, self.max_file_size],
                )
            else:
                err = "No validation set."
                logger.exception(err)
                raise ValueError(err)
        else:
            err = f"Unknown option for subset: {subset}. Use `training` or `validation`."
            logger.exception(err)
            raise ValueError(err)

    def get_class_weights(self) -> Tuple[Dict[int, float], float]:
        """
        Compute class weights and the initial bias of the model based on the seeds reserved
        as training set.

        Returns:
            Class weights dict for classes 0 and 1, along with the initial bias.
        """
        n_neg = np.count_nonzero(self.training_set[1] == 0)
        n_total = self.training_set[1].size
        n_pos = n_total - n_neg
        weight_for_uncovered = (1.0 / n_neg) * (n_total / 2.0)
        weight_for_covered = (1.0 / n_pos) * (n_total / 2.0)
        class_weights = {0: weight_for_uncovered, 1: weight_for_covered}
        initial_bias = float(np.log([n_pos / n_neg]))

        logger.info(
            "Dataset:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(
                n_total, n_pos, 100 * n_pos / n_total
            )
        )

        return class_weights, initial_bias


class CoverageSeedHandler(SeedFolderHandler):
    def __init__(
        self,
        seeds_path: pathlib.Path,
        target: List[str],
        max_len: Optional[int] = None,
        percentile_len: Optional[int] = None,
        validation_split: float = 0.0,
    ) -> None:
        super().__init__(
            seeds_path,
            target,
            bitmap_func=create_path_coverage_bitmap,
            max_len=max_len,
            percentile_len=percentile_len,
            validation_split=validation_split,
        )


def seed_data_generator(
    seed_list: List[Union[pathlib.Path, str]],
    bitmaps: np.ndarray,
    batch_size: int,
    max_seed_len: int,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Data generator based on a list of seeds and their coverage information.

    Args:
        seed_list: List of file names for seeds.
        bitmaps: Coverage bitmaps corresponding to the seeds in the list.
        batch_size: Size of batches to create.
        max_seed_len: Maximum length of seeds.

    Returns:
        Batch of data containing tuples of seeds and coverage bitmaps.
    """
    while 1:
        n_seeds = len(seed_list)
        shuffled_indices = np.random.permutation(n_seeds)

        # Load a batch of training data
        for i in range(0, n_seeds, batch_size):
            batch_indices = shuffled_indices[i : min(n_seeds, i + batch_size)]
            batch_seeds = [seed_list[seed_index] for seed_index in batch_indices]
            x = load_normalized_seeds(batch_seeds, max_len=max_seed_len)
            y = bitmaps[batch_indices]
            yield x, y


def load_normalized_seeds(seed_list: List[Union[pathlib.Path, str]], max_len: int) -> np.ndarray:
    """
    Read a batch of seeds from files, normalize and convert to Numpy array.

    Args:
        seed_list: List of paths to the seeds to read.
        max_len: Max length of seed. Longer seeds are cut, shorter ones are
            padded with zeros to reach this length.

    Returns:
        Seed content of length `max_len`, normalized between 0 and 1.
    """
    seeds = read_seeds(seed_list)

    # Pad seed with zeros up to max_len
    seeds_preproc = tf.keras.preprocessing.sequence.pad_sequences(
        seeds, padding="post", dtype="float32", maxlen=max_len
    )
    seeds_preproc = seeds_preproc.astype("float32") / 255.0

    return seeds_preproc


def read_seed(path: Union[pathlib.Path, str]) -> np.ndarray:
    """
    Read one seed from file name provided as input and return as Numpy arrays.

    Args:
        path: Path to seed to read.

    Returns:
        The content of the seed.
    """
    with open(path, "rb") as seed_file:
        seed = seed_file.read()
    return np.asarray(bytearray(seed), dtype="uint8")


def read_seeds(seed_list: List[Union[pathlib.Path, str]]) -> List[np.ndarray]:
    """
    Read multiple seeds from the list of paths provided as input and return
    them as Numpy arrays.

    Args:
        seed_list: List of paths to seeds.

    Returns:
        List of arrays of seeds.
    """
    return [read_seed(seed_file) for seed_file in seed_list]


def compute_input_size_from_seeds(
    seeds_path: pathlib.Path, percentile: int = 80, margin: int = 5
) -> int:
    """
    Compute the maximum allowed size for seeds based on a heuristic:

      * Only seeds up to the provided percentile are considered. This is to remove
        the long tail of the seed distribution.
      * An extra `margin` percent is added to the percentile value in order to still allow
        for seed growth based on inserts, splicing, etc.

    Args:
        seeds_path: Path to the seeds folder.
        percentile: Value in the 1-100 range representing the cutting point for seeds length.
        margin: Percentage in 0-100 of the percentile value to add as margin.

    Returns:
        The maximum size of a seed.
    """
    perc = seed_len_percentile(seeds_path, percentile)
    return int((1.0 + 0.01 * margin) * perc)


def seed_len_percentile(seeds_path: pathlib.Path, percentile: int = 90) -> int:
    """
    Compute the `percentile` seed length for a given input folder.

    Args:
        seeds_path: Path to the seeds folder.
        percentile: Value in the 1-100 range representing the cutting point for seeds length.

    Returns:
        The length of the seed corresponding to `percentile`.
    """
    seed_lens = [get_seed_len(seed) for seed in seeds_path.glob("*") if seed.is_file()]
    return np.percentile(seed_lens, percentile)


def get_seed_len(path: Union[pathlib.Path, str]) -> int:
    """
    Return the length of a seed based on its path.

    Args:
        path: The full path of the seed.

    Returns:
        The lengths of the seed in bytes.
    """
    return os.path.getsize(path)
