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
"""Module containing data preprocessing functions."""
import logging
import os
import pathlib
import subprocess
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CoverageBuilder:
    """
    Helper class for creating coverage labels for all seeds in a folder.
    """

    def __init__(self, target: List[str]) -> None:
        """
        Args:
            target: Target programs with args in a callable form.
        """
        # Create coverage command
        afl_path = os.environ.get("AFL_PATH", ".")
        self.command = [
            os.path.join(afl_path, "afl-showmap"),
            "-qe",
            "-o",
            "/dev/stdout",
            "-t",
            "10000",
            "-m",
            "none",
        ] + target

        # Find position in command where seed should be
        try:
            self._seed_position = self.command.index("@@")
        except ValueError:
            # If `@@` is not supplied, add it at the end of the command
            self.command.append("@@")
            self._seed_position = len(self.command) - 1

    def get_command_for_seed(self, seed: pathlib.Path) -> List[str]:
        """
        Generate the command to call for extracting the desired type of coverage
        for the seed provided as input.

        Args:
            seed: Path to seed file.

        Returns:
            Command to call in iterable format.
        """
        self.command[self._seed_position] = str(seed)
        return self.command


def create_path_coverage_bitmap(
    target_with_args: List[str],
    seed_list: List[pathlib.Path],
) -> Dict[pathlib.Path, Optional[Set[int]]]:
    """
    Create edge coverage bitmaps for each seed in `seed_list`.

    Bitmaps are extracted using the external command `afl-showmap`.
    Only edges that were already reached will be present in the bitmap.

    Args:
        target_with_args: Command line arguments used to invoke the training script.
        seed_list: List of paths containing the seeds for which bitmaps need to be extracted.

    Returns:
        Mapping of seed names to covered blocks of code in the target program.
        The covered blocks are identified via integer IDs.
    """
    logger.info("Creating edge coverage bitmaps.")
    raw_bitmap: Dict[pathlib.Path, Optional[Set[int]]] = {}
    out: bytes
    cov_tool = CoverageBuilder(target_with_args)

    has_failed_seeds = False
    for seed in seed_list:
        try:
            edges_curr_seed: Set[int] = set()
            command = cov_tool.get_command_for_seed(seed)
            out = subprocess.check_output(command)

            for line in out.splitlines():
                edge = int(line.split(b":")[0])
                edges_curr_seed.add(edge)
            raw_bitmap[seed] = edges_curr_seed
        except subprocess.CalledProcessError as err:
            raw_bitmap[seed] = None
            has_failed_seeds = True
            logger.error(f"Bitmap extraction failed: {err}")

    if has_failed_seeds:
        seed_list, raw_bitmap = _clean_seed_list(seed_list, raw_bitmap)
    if not seed_list:
        raise ValueError("No valid seed labels were produced. Stopping.")
    return raw_bitmap


def create_bitmap_from_raw_coverage(
    raw_bitmap: Dict[pathlib.Path, Set[int]]
) -> Tuple[List[pathlib.Path], np.ndarray]:
    """
    Given raw coverage information for seeds, create a compact and compressed
    coverage bitmap.

    Args:
        raw_bitmap: Mapping of seed names to covered blocks in the target program.
            The covered blocks are identified via integer IDs.

    Returns:
        * Ordered seed list corresponding to the compressed bitmap.
        * Numpy array containing the "reduced" coverage bitmap for all seeds.
        The bitmap is reduced by merging together the edges (columns) that have identical coverage
        under the existing seeds.
    """
    seed_list = list(raw_bitmap.keys())
    all_edges = set.union(*raw_bitmap.values())
    all_edges_indices = {addr: index for index, addr in enumerate(all_edges)}
    cov_bitmap = np.zeros((len(seed_list), len(all_edges)), dtype=bool)
    for seed_idx, seed in enumerate(seed_list):
        for addr in raw_bitmap[seed]:
            cov_bitmap[seed_idx][all_edges_indices[addr]] = True
    del all_edges, all_edges_indices

    reduced_bitmap = remove_identical_coverage(cov_bitmap)
    assert len(seed_list) == reduced_bitmap.shape[0]
    return seed_list, reduced_bitmap


def remove_identical_coverage(bitmap: np.ndarray, keep_unseen: bool = False) -> np.ndarray:
    """
    Reduce a coverage bitmap by merging together all edge coverage with identical
    coverage under all seeds.

    Args:
        bitmap: Boolean coverage bitmap, where each row is a seed and each column an edge.
        keep_unseen: True if unseen addressed or edges (columns of zeroes) should be left intact.

    Returns:
        Reduced bitmap.
    """
    if keep_unseen:
        mask = np.sum(bitmap, axis=0) == 0
        _, ind_unique_cov = np.unique(bitmap, axis=1, return_index=True)
        mask[ind_unique_cov] = 1
        reduced_bitmap = bitmap[:, mask]
    else:
        reduced_bitmap = np.unique(bitmap, axis=1)

    logger.info(f"Bitmap reduced from {str(bitmap.shape)} to {str(reduced_bitmap.shape)}")
    return reduced_bitmap


def _clean_seed_list(seed_list, raw_bitmap):
    logger.info("Removing failed seeds from dataset.")
    n_removed = 0
    for seed in reversed(seed_list):
        if raw_bitmap[seed] is None:
            del raw_bitmap[seed]
            seed_list.remove(seed)
            n_removed += 1

    logger.info(f"Successfully removed {n_removed} seeds.")
    assert len(seed_list) == len(
        raw_bitmap
    ), f"The number of seeds and labels do not match: {len(seed_list)}, {len(raw_bitmap)}"
    return seed_list, raw_bitmap
