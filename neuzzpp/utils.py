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
"""Module containing diverse utility functions."""
import logging
import pathlib
import subprocess
import time
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

logger = logging.getLogger(__name__)


class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    """Custom TensorBoard callback that tracks the learning rate."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr_schedule = getattr(self.model.optimizer, "lr", None)
        if callable(lr_schedule):
            val = lr_schedule(self.model.optimizer.iterations)
        elif lr_schedule is not None:
            val = lr_schedule
        logs.update({"lr": tf.keras.backend.eval(val)})
        super().on_epoch_end(epoch, logs)


def model_needs_retraining(
    seeds_path: pathlib.Path,
    timestamp_last_training: int,
    n_seeds_last_training: int,
    retraining_interval_s: int = 3600,
    n_new_seeds_for_retraining: int = 10,
) -> bool:
    """
    Function determining if the machine learning model needs retraining based on two criteria:
      * Time elapsed since the model was trained (if ever).
      * No. of new seeds found since last training.

    Args:
        seeds_path: Path to the seeds folder.
        timestamp_last_training: Unix timestamp of last training.
        n_seeds_last_training: No. of seeds in the corpus at the last training time.
        retraining_interval_s: Minimal interval between model training rounds, in seconds.
        n_new_seeds_for_retraining: Minimal no. of new seeds necessary to trigger training.

    Returns:
        True if the model should be retrained.
    """
    n_current_seeds = len(list(seeds_path.glob("id*")))
    time_since_retrain = int(time.time()) - timestamp_last_training
    return (
        time_since_retrain >= retraining_interval_s
        and n_current_seeds >= n_seeds_last_training + n_new_seeds_for_retraining
    )


def get_max_file_size(path: Union[pathlib.Path, str]) -> int:
    """
    Returns the maximum file size in the given path.

    The folder is *not* scanned recursively.

    Args:
        path: Folder path to read.

    Returns:
        The size of the largest file.
    """
    files = pathlib.Path(path).glob("*")
    return max([file.stat().st_size for file in files])


def create_work_folders(path: Union[str, pathlib.Path] = ".") -> None:
    """
    Create folder for machine learning models.

    Args:
        path: Path where to create work folders.
    """
    folders_to_create = ["models"]
    parent = pathlib.Path(path) if isinstance(path, str) else path
    for folder in folders_to_create:
        (parent / folder).mkdir(parents=True, exist_ok=True)


def get_timestamp_millis_from_filename(filename: str) -> int:
    """
    Extracts the timestamp that AFL++ encodes into the filenames of the queue files.

    Args:
        filename: Full path to the seed.
    """
    for token in filename.split(","):
        key_val = token.split(":")
        if key_val[0] == "time":
            return int(key_val[1])

    return 0


def _add_to_dict(data_dict, key, value):
    """Append value to its corresponding key in the dict without erasing existing values."""
    if key in data_dict:
        data_dict[key].append(value)
    else:
        data_dict[key] = [value]


def _search_afl_plot_data(
    folder: pathlib.Path, data_columns: List[str], plot_file: str
) -> Dict[str, pd.DataFrame]:
    """
    Search input folder for AFL++ plotting data. The last folder in the path containing the plot
    data file will be considered an experiment trial. If multiple trials are available, they will
    be aggregated into one experiment and a confidence band will be computed.
    """
    all_plot_data: Dict[str, pd.DataFrame] = {}
    all_plot_data_files = list(folder.glob(f"**/{plot_file}"))

    for plot_data in all_plot_data_files:
        cov_data = pd.read_csv(plot_data, sep=", ", usecols=data_columns, engine="python")
        if "default" in str(plot_data):
            experiment_key = str(plot_data.relative_to(folder).parent.parent.parent)
        else:
            experiment_key = str(plot_data.relative_to(folder).parent.parent)
        _add_to_dict(all_plot_data, experiment_key, cov_data)

    return all_plot_data


def create_plot_afl_coverage(
    folders: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]], plot_file: str = "plot_data"
):
    """
    Create and return a plot displaying coverage over time data extracted from the input folders.

    This function searches all subpaths of input folders for AFL/AFL++ coverage results `plot_data`,
    or another type of plot file specified by `plot_file`. For experiments running multiple trials,
    their average is plotted with 95% confidence intervals.

    The returned plot can be displayed with `matplotlib.pyplot.show` or saved with
    `matplotlib.pyplot.savefig`. As it is a plot object, its appearance can be changed
    (e.g., title, labels, colors) using standard functions.

    Args:
        folders: Folder path or list of folder paths to search for plotting results.
        plot_file: Name of the file containing plot data.

    Returns:
        Plot object returned by `seaborn` / `matplotlib`.
    """
    sns.set_theme()
    data_columns = ["edges_found"]
    time_column_aflpp = "# relative_time"
    time_column_afl = "relative_time"
    all_plot_data: Dict[str, pd.DataFrame] = {}

    if not isinstance(folders, list):
        folders = [folders]

    # Walk folders and read plot data
    for folder in folders:
        path = pathlib.Path(folder).expanduser()
        try:
            all_plot_data.update(
                _search_afl_plot_data(
                    path, data_columns + [time_column_aflpp], plot_file=plot_file
                )
            )
        except ValueError:
            all_plot_data.update(
                _search_afl_plot_data(
                    path, data_columns + [time_column_afl], plot_file=plot_file
                )
            )

    # Preprocess data from each trial in preparation for merging
    for trials in all_plot_data.values():
        for i, trial in enumerate(trials):
            # Rename all timestamp columns to AFL++ name
            if time_column_afl in trial.columns:
                trial.rename(columns={time_column_afl: time_column_aflpp}, inplace=True)

            # Fill missing values
            idx = np.arange(1, max(86400, trial[time_column_aflpp].max()) + 1)
            trial = trial.set_index(time_column_aflpp)
            trial = trial[~trial.index.duplicated()]
            trial = trial.reindex(idx).reset_index()
            trial.ffill(inplace=True)

            # Keep only 1/900 values, as it does not degrade plot quality
            if len(idx) > 900:
                trial = trial.loc[trial[time_column_aflpp] % 900 == 0]
            trials[i] = trial

    # Merge trials of same experiment in long format
    for exp, trials in all_plot_data.items():
        if len(trials) > 1:
            all_plot_data[exp] = pd.concat(trials, ignore_index=True, sort=False)
        else:
            all_plot_data[exp] = trials[0]

    # Merge experiments in long format
    plot_data_df = None
    if len(all_plot_data) > 1:
        # Merge results from multiple fuzzers
        plot_data_df = pd.concat(all_plot_data.values(), keys=list(all_plot_data.keys()))
        plot_data_df.reset_index(level=0, inplace=True)
        plot_data_df.reset_index(drop=True, inplace=True)

    else:
        # Only one fuzzer in experiments
        for exp, trials in all_plot_data.items():
            plot_data_df = trials
            plot_data_df["level_0"] = exp

    if plot_data_df is not None:
        # Rename columns for plotting
        plot_data_df.rename(
            columns={
                "level_0": "Fuzzer",
                "# relative_time": "Relative time (hours)",
                "edges_found": "Edge coverage",
            },
            inplace=True,
        )

        # Create plot
        plot = sns.lineplot(
            data=plot_data_df, x="Relative time (hours)", y="Edge coverage", hue="Fuzzer"
        )
        plot.legend(loc="lower right")
        plot.set_title("Average edge coverage over time", fontsize=24)
        plot.set_xticks(range(0, 3600 * 25, 3600 * 6))
        plot.set_xticklabels([str(x) for x in range(0, 25, 6)])

        return plot


def _read_last_line_csv(path: pathlib.Path, columns: List[str]) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as plot_file:
        total_cov = pd.read_csv(plot_file, sep=", ", usecols=columns, engine="python").iloc[-1]

    return total_cov


def compute_coverage_experiment(folder: Union[str, pathlib.Path]) -> pd.DataFrame:
    """
    Extract coverage information from an experiment into a Pandas dataframe.

    Assumptions:
      * Coverage files are called `replayed_plot_data`.
      * An experiment is structured as:
          <exp_name>/<target>/<fuzzer>/trial-<index>/<fuzzer_output>
        or
          <exp_name>/<target>/<fuzzer>/trial-<index>/default/<fuzzer_output>.
      * The name of the plot data column used for computations is "edges_found".

    Args:
        folder: Experiment folder structured as specified above.

    Returns:
        Dataframe containing for each target and fuzzer:
          * The average edge coverage for all trials.
          * The standard deviation of edge coverage.
    """
    data_columns = ["edges_found"]

    # Walk folders and read plot data
    coverage_info = dict()
    if isinstance(folder, str):
        folder = pathlib.Path(folder).expanduser()
    for target in folder.glob("*"):
        for fuzzer in target.glob("*"):
            total_cov_trials = []
            for trial in fuzzer.glob("trial-*"):
                plot_files = list(trial.glob("**/replayed_plot_data"))
                assert len(plot_files) == 1
                plot_file = plot_files[0]

                # Read total coverage
                total_cov = _read_last_line_csv(plot_file, data_columns)
                total_cov_trials.append(total_cov)
            coverage_info[(target.name, fuzzer.name)] = [
                int(np.mean(total_cov_trials)),
                np.std(total_cov_trials),
            ]

    cov_pd = {
        "index": list(coverage_info.keys()),
        "index_names": ["target", "fuzzer"],
        "columns": [
            "Avg. edge cov.",
            "Std. dev.",
        ],
        "column_names": ["metrics"],
        "data": list(coverage_info.values()),
    }

    return pd.DataFrame.from_dict(cov_pd, orient="tight")


def replay_corpus(out_path: pathlib.Path, target: pathlib.Path):
    """
    Script to replay the fuzzing corpus from the `queue` folder in `out_path`.
    The replay is done using AFL++ `afl-showmap` on targets built for AFL.

    Args:
        out_path: Output folder of the fuzzing run.
        target: Path of the fuzzing target.

    Raises:
        `subprocess.CalledProcessError`: When the corpus replay fails.
    """
    try:
        subprocess.run(
            [
                pathlib.Path("/mlfuzz") / "scripts" / "replay_corpus.py",
                out_path / "queue",
                out_path / "replayed_plot_data",
                target.parent / (target.stem + ".afl"),
            ]
        )
    except subprocess.CalledProcessError as err:
        logger.warning(f"{err.output}. Skipping corpus replay.")
        print(f"{err.output}. Skipping corpus replay.")


def kill_fuzzer(fuzzer_command: str = "afl-fuzz", output_stream=subprocess.DEVNULL):
    """
    Kill a fuzzing process by name.

    This command is necessary to stop AFL-based fuzzers after a given time.

    Args:
        fuzzer_command: Name of the process to kill.
        output_stream: Stream for redirecting `stdout` and `stderr` of the kill command.
    """
    # Can't avoid this because 'run_afl_fuzz' doesn't return a handle to
    # 'afl-fuzz' process so that we can kill it with subprocess.terminate()
    subprocess.call(["pkill", "-f", fuzzer_command], stdout=output_stream, stderr=output_stream)
