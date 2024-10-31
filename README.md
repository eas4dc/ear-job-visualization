# ear-job-visualization

A tool to automatically read and visualise runtime data provided by the [EAR](https://gitlab.bsc.es/ear_team/ear/-/wikis/home) software.
**ear-job-visualization** is a cli program written in Python which lets you plot the EAR data given by some of its commands or by using some report plug-in offered by the EAR Library (EARL).
The main visualisation target is to show runtime metrics collected by the EAR Library in a timeline graph.

By now this tool supports two kind of output formats:
1. Directly generate images showing runtime information.
2. Generate a trace file to be read by Paraver, a tool to visualise and manage trace data maintaned by the Barcelona Supercomputing Center's Tools team.

For more information, read about [eacct](https://gitlab.bsc.es/ear_team/ear/-/wikis/EAR-commands#ear-job-accounting-eacct) or [this guide](https://gitlab.bsc.es/ear_team/ear/-/wikis/User%20guide#running-jobs-with-ear) which shows you how to run jobs with EAR and how to obtain runtime data.
You can find [here](https://tools.bsc.es/paraver) more information about how Paraver works.

## Features

- Generate static images showing runtime metrics of your job monitored by EARL.
- Generate Paraver traces to visualize runtime metrics within Paraver tool or any other tool of the BSC's Tools teams.
- **(Beta)** Generate a LaTeX project with the most relevant information about the job to be analyzed.
    - Job global summary.
    - Job phase classification.
    - Job runtime metrics.

## Requirements

- pandas
- matplotlib
- importlib_resources

~By default, the tool calls internally the EAR account command (i.e., *eacct*) with the proper information and options in order to get the corresponding data to be sent to the tool's functionalities.~
> ~Be sure you have the the *eacct* command on your path, and also check whether `EAR_ETC` environment variable is set properly. By loading `ear` module you should have all the needed stuff ready.~

~If you have some trouble, ask your system administrator if there is some problem with the EAR Database.
You can also provide directly input files if eacct is unable, [read below](https://github.com/eas4dc/ear-job-analytics/blob/main/README.md#providing-files-instead-of-using-internally-eacct).~

## Installation

This repository contains all recipes to build and install the package.
You need **build** and **setuptools** packages to properly build and install this one.

```bash
pip install -U pip
pip install build setuptools
python -m build
pip install .
```

Tool's developers may want to use `pip install -e .` to install the package in editable mode, so there is no need to reinstall every time you want to test a new feature.

Then, you can type `ear-job-analytics` and you should see the following:

```
usage: ear-job-visualization [-h] [--version] [-c CONFIG_FILE] (--format {runtime,ear2prv,summary} | --print-config | --avail-metrics)
                         [--input-file INPUT_FILE] [-j JOB_ID] [-s STEP_ID]
                         [-o OUTPUT] [-k] [-t TITLE] [-r]
                         [-m metric [metric ...]]
ear-job-analytics: error: one of the arguments --format --print-config --avail-metrics is required
```

If you had some trouble during the build and/or installation process, contact to support@eas4dc.com.
We are trying provide a more easy way to install the package.

### Make the package usable by other users

You can install the tool to be available to other users in multiple ways, and maybe you know a better approach for doing so or which fits much better to your use case, but here there is explained a way we found useful to fit on systems where we put this tool in production.

1 - Prepend the path to `site-packages` directory where you have installed the tool to `PYTHONPATH`.
2 - Prepend the path to `bin` directory where you have installed the tool to `PATH`.

For example, if you have installed the tool in a virtual environment located in a directory where other users have read and execute permissions, you may want to provide users a module file which prepends `virtualenv/install/dir/lib/python<version>/site-packages` to `PYTHONPATH` variable and `virtualenv/install/dir/bin` to `PATH`.

```lua
# An example module file for Lmod

whatis("Enables the usage of ear-job-analytics, a tool for visualizing performance metrics collected by EAR.")

prepend_path("PYTHONPATH", "virtualenv/install/dir/lib/python<version>/site-packages")
prepend_path("PATH", "virtualenv/install/dir/bin")
```

## Usage

You must choose one of the three main required options.
The one you may use most of times is `--format`, but the order followed in this document is useful for new users to understand how the tool works.

### `--print-config`

Pretty prints the [configuration](Configuration) being used by the tool.
You can take the printed configuration as an example for making yours and use it later through `--config-file` option.
The usage of this flag is very simple:

```bash
ear-job-visualization --print-config
```

### `--avail-metrics`

Shows metrics names supported by tool.
These supported metrics are taken from the configuration file, so you can view the default supported metrics with:

```bash
ear-job-analytics --avail-metrics
```

You can also check you own configuration file:

```bash
ear-job-analytics --avail-metrics -c my_config.json
```

### `--format`

This option is in fact used to request for plotting (or converting) data.

Choices for this option are either *runtime*, *ear2prv* ~or *job-summary (beta)*~, and each one enables each of the tool's features.
Read below sections for a detailed description of each one.

The *runtime* option is the one used to generate static images, while *ear2prv* refers the tool's interface to output data following the Paraver Trace Format.
Finally, *job-summary* generates an overview analysis of the most relevant information of the job.

> _job-summary_ format is not in production yet.

You must use the `--input-file` option to specify where the tool will find the data.
You need two files:

- Loop csv file: EAR loop signature CSV file obtained by using `--ear-user-db` flag when launching an application. Normally named `<app name>.<node name>.time.loops.csv`.
- Job csv file: EAR application global signatures obtained by using `--ear-user-db` flag (both files are generated by setting the flag just once). Normally named `<app name>.<node name>.time.csv`.

**You must rename the job csv file** in order to get the tool working. Set the same name as loop csv file and prepend the string `out_jobs.`:

```bash
mv <app name>.<node name>.time.csv out_jobs.<app name>.<node name>.time.loops.csv
```

Put both files in the same path, and pass the original loop csv file name to the tool (i.e., `--input-file <app name>.<node name>.time.loops.csv`).

Finally, you must specify the Job ID (i.e., `--job-id`) of the job being analyzed as features currently only support working with data corresponding with one Job.
This required option ensures the tool to filter the input data by Job ID to avoid possible errors on the output.
So, the minimum shape of an invokation is:

```bash
ear-job-analytics --format [runtime|ear2prv|summary] --input-file <loops csv file> --job-id <JobID>
```

### ~Providing files instead of using internally eacct~

~If you know which *eacct* invokations are required to visualise the data, you can use the option *--input-file* to specify where the tool will find the data to be filtered by the two required job-related options (e.g., *--job-id*, *--step-id*).
This option is useful when you already have data for multiple jobs and/or steps together and you want to work on it in several ways because naturally it's more fast to work directly on a file than invoking a command to make a query to a Database, storing the output on a file, and then read such file.
This option is also useful since it lets you work on a host where you can't access EAR Database nor EAR is installed.~

~The way how the value of this option is handled depends on which functionality (e.g., *format*) you are working on, and which kind of data you want to produce/visualise.
If **runtime** format option is used, the *--input-file* option can be a single filename (which can be given with its relative path) wich contains EAR loop data.~
~~If a directory name is given, the tool will read all files inside it (another reason why it is required to specify the Job and Step IDs).~~

> ~If you started working by using *eacct* command internally, all required files are stored temporally while the tool is doing its work.~
> ~If you want to reuse such files later you can pass the option `--keep-csv` to prevent files been removed.~
> ~Then, you can provide those files to get different output.~

### *runtime* format

Generate a heatmap-based graph for each metric specified by `--metrics` argument (i.e., space separated list of metric names).
Note that the accepted metrics by your **ear-job-analytics** installation are specified in the [configuration](Configuration) file and you can request the list trough `--avail-metrics` flag.

The resulting figure (for each metric specified) will be a timeline where for each node your application had used you will see a heatmap showing an intuitive visualisation about the value of the metric during application execution.
All nodes visualised share the same timeline, which makes this command useful to check the application behaviour over all of them.

#### Examples

```
$> ear-job-analytics --format runtime --input-file test_files/loops.gromacs_223676.csv -j 223676 -s 0 --save -l -r -m dc_power
reading file test_files/loops.gromacs_223676.csv
storing figure runtime_dc_power-223676-0
```

You can check the resulting figure [here](src/extra/examples/imgs/runtime_dc_power-223676-0.pdf).

#### Request metrics

The *--metrics* option allows you request one or more metrics to visualize.
Type `ear-job-analytics --help` to see which of them are currently available.
Metrics are specified by a configuration file (not documented yet), so it's easy to extend the supported ones.
Contact with support@eas4dc.com to request more metrics.

By default, the range to compute each metric runtime gradient is configured at *config.json*, but you can tell the tool to compute the gradient based on the range of the current data by typing `--relative-range` option before requestingthe metrics list:

```
$> ear-job-analytics --format runtime --input-file test_files/loops.gromacs_223676.csv --job-id 223676 --step-id 0 --relative-range -m dc_power cpi
```

#### Change the output

By default, the legend of the gradient is displayed vertically at the right of timelines.
If there are a few timelines, the height of the legend can be too short to correctly visualize the color range legend.
You can set the `--horizontal-legend` option to display the legend horizontally below timelines, so you make sure the size is sufficient to read color codes.

## Configuration

Check the [config.json](src/ear_analytics/config.json) file.

## Contact

For any question and suggestion, contact with support@eas4dc.com.
You can also open an issue in this repository.
