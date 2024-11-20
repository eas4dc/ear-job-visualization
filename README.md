# ear-job-visualize

A tool to automatically read and visualise runtime data provided by the [EAR](https://gitlab.bsc.es/ear_team/ear/-/wikis/home) software.
**ear-job-visualize** is a cli program written in Python which lets you plot the EAR data given by some of its commands or by using some report plug-in offered by the EAR Library (EARL).
The main visualisation target is to show runtime metrics collected by the EAR Library in a timeline graph.

By now this tool supports two kind of output formats:
1. Directly generate images showing job runtime information.
2. Generate a trace file to be read by Paraver, a tool to visualise and manage trace data maintaned by the Barcelona Supercomputing Center's Tools team.

For more information, read about [eacct](https://gitlab.bsc.es/ear_team/ear/-/wikis/EAR-commands#ear-job-accounting-eacct) or [this guide](https://gitlab.bsc.es/ear_team/ear/-/wikis/User%20guide#running-jobs-with-ear) which shows you how to run jobs with EAR and how to obtain runtime data.
You can find [here](https://tools.bsc.es/paraver) more information about how Paraver works.

## Features

- Generate static images showing runtime metrics of your job monitored by EARL.
- Generate Paraver traces to visualize runtime metrics within Paraver tool or any other tool of the BSC's Tools teams.

## Requirements

- pandas
- matplotlib
- importlib\_resources

By default, the tool calls internally the EAR account command (i.e., `eacct`) with proper options in order to get the corresponding data to be sent to the tool's functionalities.
> Be sure you have the the `eacct` command on your path, and also check whether `EAR_ETC` environment variable is set properly. By loading `ear` module you should have all the needed stuff ready.

If you have some trouble, ask your system administrator if there is some problem with the EAR Database.
You can also provide directly input files if the `eacct` command is unable, [read below](#providing-files-instead-of-using-internally-eacct).

## Installation

This repository contains all recipes to build and install the package.
You need **build** and **setuptools** packages to properly build and install it.
You can also use the [`PYTHONUSERBASE`](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONUSERBASE) environment variable to modify the target directory for the installation.

```bash
pip install -U pip
pip install build setuptools
python -m build
pip install .
```

> You can change the destination path by export the variable 
> Tool's developers may want to use `pip install -e .` to install the package in editable mode, so there is no need to reinstall every time you want to test a new feature.

Then, you can type `ear-job-visualizer` and you should see the following:

```
usage: ear-job-visualizer [-h] [--version] [-c CONFIG_FILE]
                          (--format {runtime,ear2prv} | --print-config | --avail-metrics)
                          [--loops-file LOOPS_FILE] [--apps-file APPS_FILE]
                          [-j JOB_ID] [-s STEP_ID] [-o OUTPUT] [-k] [-t TITLE]
                          [-r] [-m metric [metric ...]]
ear-job-visualizer: error: one of the arguments --format --print-config --avail-metrics is required
```

### Make the package usable by other users

You can install the tool to be available to other users in multiple ways, and maybe you know a better approach for doing so or which fits much better to your use case, but here there is explained a way we found useful to fit on systems where we put this tool in production.

1. Export the [`PYTHONUSERBASE`](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONUSERBASE) environment variable to modify the target directory for the installation.
2. Prepend the path to `site-packages` directory where you have installed the tool to `PYTHONPATH`.
3. Prepend the path to `bin` directory where you have installed the tool to `PATH`.

For example, if you have installed the tool in a virtual environment located in a directory where other users have read and execute permissions, you may want to provide users a module file which prepends `virtualenv/install/dir/lib/python<version>/site-packages` to `PYTHONPATH` variable and `virtualenv/install/dir/bin` to `PATH`.

```lua
# An example module file for Lmod

whatis("Enables the usage of ear-job-visualizer, a tool for visualizing performance metrics collected by EAR.")

depends_on("") # Add here the required python module you used for building the package.
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
ear-job-visualization --print-config > my_config.json
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

Choices for this option are either `runtime`, `ear2prv`, and each one enables each of the tool's features.
Read below sections for a detailed description of each one.

The `runtime` option is the one used to generate static images, while `ear2prv` refers the tool's interface to output data following the Paraver Trace Format.
Both format options share a subset of arguments.

The `--job-id` flag is **mandatory** to be specified.
It is used by the tool to filter input data in the case it contains more than one Job ID, as it currently only supports single job visualisation.
Moreover, you can set the `--step-id` flag to filter also the Step ID, which is **mandatory for `--format runtime` option** and **optional for `--format ear2prv`**, since the latter supports multiple step data in the input.

By default, the tool will internally call the `eacct` command and will store the data into temporary files.
Those files will be used by the tool and are removed at the end.
If you want to prevent the removal of that files, you can add the `--keep-csv` flag.

If you know which `eacct` invokations are required to visualise the data, you can use `--loops-file` and `--apps-file` options to specify where the tool can find the data to be filtered and used.
**Both of them are required if you are going to use the tool escaping the internal use of the `eacct` command.**
The former is obtained through `eacct -j <jobid>[.stepid] -r -c <loops_file>` and the latter through `eacct -j <jobid>[.stepid] -l -c <apps_file>`.
You can alternatively obtain both files by using one of the EAR [report plug-ins](https://gitlab.bsc.es/ear_team/ear/-/wikis/Report#csv) distributed with EAR.
This option is useful when you already have data for multiple jobs and/or steps together and you want to work on it in several ways because naturally it's more fast to work directly on a file than invoking a command to make a query to a Database, storing the output on a file, and then read such file.
This option is also useful since it lets you work on a host where you can't access EAR Database nor EAR is installed.

### *runtime* format

Generate a heatmap-based graph for each metric specified by `--metrics` argument (i.e., space separated list of metric names).
Note that the accepted metrics are specified in the [configuration](Configuration) file and you can request the list trough the `--avail-metrics` flag.

> This option just supports plotting data for a single Job-Step ID with just one Application ID (i.e., non-workflow use case),
> so both `--job-id` and `--step-id` flags are required.

The resulting figure (for each requested metric) will be a timeline where for each node your application had used you will see a heatmap showing an intuitive visualisation about the value of the metric during application execution.
All nodes visualised share the same timeline, which makes this command useful to check the application behaviour over all of them.
If you request GPU metrics, the graph will show you per-GPU data.

By default, the range to compute each metric runtime gradient is configured at *config.json*, but you can tell the tool to compute the gradient based on the range of the current data by typing `--relative-range` option before requestingthe metrics list.

### *ear2prv* format

Convert job runtime data gathered from EARL to Paraver Trace Format.
In this case, all metrics found in the input data are reported to the trace file.
Moreover, you can have in the same trace all steps and applications (e.g., a workflow) of your job, so just the `--job-id` flag is required.

You can find two examples of [Paraver Config Files](examples) to easily start working with the output data generated by this option.

## Configuration

Check the [config.json](src/ear_analytics/config.json) file.

## Contact

For any question and suggestion, contact with support@eas4dc.com.
You can also open an issue in this repository.
