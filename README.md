# Neuzz++ - Neural program smoothing for fuzzing in AFL++

Neuzz++ is an implementation of neural program smoothing for fuzzing as AFL++ custom mutator.
The mutator is implemented in C and interacts with Python for training a machine learning model and generating mutations.
This is the companion code for the Neuzz++ method reported in the paper ["Revisiting Neural Program Smoothing for Fuzzing"](https://arxiv.org/abs/2309.16618) presented at ESEC/FSE'23.
See also the [MLFuzz benchmarking framework](https://github.com/boschresearch/mlfuzz) introduced in the same paper.

## Installation

### Installing AFL++

Neuzz++ is implemented as a custom mutator for AFL++, so it requires this fuzzer to be installed.
For reproducing experimental results from the paper, we recommend using the AFL++ version specified by the commit hash below.
We provide two alternative installation options:
* Either clone and compile AFL++ from source in the folder of your choice:

      git clone https://github.com/AFLplusplus/AFLplusplus
      cd AFLplusplus/
      git checkout 9e2a94532b7fd5191de905a8464176114ee7d258
      make

* Or install from Ubuntu repositories:

      sudo apt install afl++

### Install Python dependencies

This project uses `python>=3.8` and [`poetry`](https://python-poetry.org/) for managing the Python environment.
Install `poetry` system-wide or in an empty virtual environment (e.g., created via `virtualenv` or `conda`).
Then run

    poetry install --without dev

to install the project dependencies.
Note that Neuzz++ and MLFuzz have the same Pythhon dependencies; you only need to create one virtual environment for both of them.
Use

    poetry shell

to activate the environment.

### Build Neuzz++ custom mutator

In the cloned `NEUZZplusplus` folder, run:

    make -C ./aflpp-plugins/

### Set environment variables

Finally, export the `AFL_PATH` and `NEUZZPP_PATH` pointing to the cloned repos:

    export AFL_PATH=/path/to/AFLplusplus/
    export NEUZZPP_PATH=/path/to/NEUZZplusplus/

You are now ready to use Neuzz++.

## Usage

### Basic usage

Running Neuzz++ is done by running AFL++'s fuzzing command `afl-fuzz` with the custom mutator environment variable `AFL_CUSTOM_MUTATOR_LIBRARY` pointing at the library built in previous steps (`./aflpp-plugins/libml-mutator.so`).
All standard AFL++ options can be used in conjunction with Neuzz++; please see the AFL++ [official page](https://aflplus.plus/) for more information on these and on building targets for fuzzing.
In the following, we provide a basic usage example.

**Note** Neuzz++ requires enabling the AFL++ `AFL_DISABLE_TRIM=1` option (e.g., by using environment variables).

In the Neuzz++ base folder, activate the Python virtual environment by calling:

    poetry shell

or the appropriate command for your environment management tool.
To fuzz a target program called `target` available in the current folder, run:

    AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 AFL_FORKSRV_INIT_TMOUT=1000 AFL_DISABLE_TRIM=1 AFL_CUSTOM_MUTATOR_LIBRARY=./aflpp-plugins/libml-mutator.so \
    afl-fuzz -i input -o output -m none -- ./target

If any valid inputs for the target program are available (also known as *seeds*), place them in the `input` folder.
The results of the fuzzing campaign will be stored in the `output` folder.
The fuzzed program does not have to be in the current folder; just use the correct relative path.
Note that the target needs to be built with [AFL++ instrumentation](https://aflplus.plus/docs/fuzzing_in_depth/#1-instrumenting-the-target) for fuzzing.
To run the experiment for a fixed duration, use the `-V` option, followed by the duration in seconds (e.g., `-V 86400` will fuzz for 24 hours).
The `input`, `output` and `target` paths can be customized to the user's needs.

The maximum number of gradients that should be used for mutations can be set via the environment variable `NEUZZPP_MAX_GRADS`.
The default value is `NEUZZPP_MAX_GRADS=32`; this value is used in the paper and is validated experimentally for a good performance-speed trade-off.

### Framework usage

Neuzz++ is integrated with the open-source fuzzing framework [MLFuzz](../MLFuzz/README.md), which allows to run containerized, large-scale fuzzing experiments on standard target programs.
Please see MLFuzz [setup](../MLFuzz/README.md#setup) and [usage instructions](../MLFuzz/README.md#usage) for more details.

### Fuzzing output

At the end of a Neuzz++ fuzzing run, the `output` folder will contain the following information:

    output
    ├── crashes           # Folder that containing inputs that trigger crashes on the target
    ├── fuzzer_config     # AFL++ config for this run
    ├── fuzzer_stats      # Statistics of the fuzzing campaign
    ├── models            # Folder storing machine learning models if the option was enabled in the Python script
    ├── plot_data         # CSV data regarding fuzzing progression and coverage over (relative) time
    └── queue             # Corpus of all interesting inputs found by the fuzzer

## Project structure

Neuzz++ follows a standard Python package structure.

    NEUZZplusplus/
    ├── aflpp-plugins/          # Neuzz++ custom mutator for AFL++ linking to Python code for ML 
    ├── docs/                   # Sphinx documentation sources
    ├── neuzzpp/                # Python package with reusable ML logic
    ├── notebooks/              # Jupyter notebooks reproducing mutations effectiveness analysis
    ├── scripts/                # Scripts folder with Neuzz++ ML code used by AFL++ custom mutator
    ├── LICENSE                 # License file
    ├── poetry.lock             # Project requirements in Poetry format
    ├── pyproject.toml          # Standard Python package description for pip
    └── README.md               # The present README file

## Citation

If you use Neuzz++ in scientific work, consider citing our paper presented at ESEC/FSE '23:

    Maria-Irina Nicolae, Max Eisele, and Andreas Zeller. “Revisiting Neural Program Smoothing for Fuzzing”. In Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering. ACM, Dec. 2023.

<details>
<summary>BibTeX</summary>

  ```bibtex
  @inproceedings {NEUZZplusplus23,
  author = {Maria-Irina Nicolae, Max Eisele, and Andreas Zellere},
  title = {Revisiting Neural Program Smoothing for Fuzzing},
  booktitle = {Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE)},
  year = {2023},
  publisher = {{ACM}},
  doi = {10.1145/3468264.3473932},
  month = dec,
  }
  ```

</details>

## License

Copyright (c) 2023 Robert Bosch GmbH and its subsidiaries.
Neuzz++ is distributed under the AGPL-3.0 license.
See the [LICENSE](LICENSE) for details.
