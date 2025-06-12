# ear-job-visualizer

A tool to automatically read and visualise runtime data provided by the [EAR](https://gitlab.bsc.es/ear_team/ear/-/wikis/home) software.
**ear-job-visualizer** is a cli program written in Python which lets you plot the EAR data given by some of its commands or by using some report plug-in offered by the EAR Library (EARL).
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
- rich

By default, the tool calls internally the EAR account command (i.e., `eacct`) with proper options in order to get the corresponding data to be sent to the tool's functionalities.
> Be sure you have the the `eacct` command on your path, and also check whether `EAR_ETC` environment variable is set properly. By loading `ear` module you should have all the needed stuff ready.

If you have some trouble, ask your system administrator if there is some problem with the EAR Database.
You can also provide directly input files if the `eacct` command is unable, [read below](#providing-files-instead-of-using-internally-eacct).

## Installation

Clone the repository with its submodules, since it depends on the [ear\_analytics\_core](https://github.com/eas4dc/ear_analytics_core):

```bash
git clone --recurse-submodules git@github.com:eas4dc/ear-job-visualization.git
```

If you already cloned this repository without using `--recurse-submodules` flag, you need to manually install the [ear\_analytics\_core](https://github.com/eas4dc/ear_analytics_core) repository into the `src` directory, located at the root of this one.

If you do not have internet access you can clone this repository and its dependencies with the following commands:

```bash
git clone --bare git@github.com:eas4dc/ear-job-visualization.git
git clone --bare https://github.com/eas4dc/ear_analytics.git
```
After scp your bare repositories to the machine, and:

```bash
git clone ear-job-visualization.git
git clone ear_analytics.git
cd ear-job-visualization/src
rmdir ear_analytics
ls -s ../../ear_analytics
```

This repository contains all recipes to build and install the package.
You need **build** and **setuptools** packages to properly build and install it.
You can also use the [`PYTHONUSERBASE`](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONUSERBASE) environment variable to modify the target directory for the installation.

```bash
pip install -U pip
pip install build setuptools wheel
python -m build
pip install .
```

> You can change the destination path by exporting the variable [`PYTHONUSERBASE`](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONUSERBASE).
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

For example, if you have installed the tool in a virtual environment located in a directory where other users have read and execute permissions, you may want to provide users a module file which prepends `<prefix>/lib/python<version>/site-packages` to `PYTHONPATH` variable and `<prefix>/bin` to `PATH`[^2]. You can use the python script create_module.py to generate the module file. 

```lua
# An example module file for Lmod

whatis("Enables the usage of ear-job-visualizer, a tool for visualizing performance metrics collected by EAR.")

-- Add here the required python module you used for building the package.
-- depends_on("")
prepend_path("PYTHONPATH", "virtualenv/install/dir/lib/python<version>/site-packages")
prepend_path("PATH", "virtualenv/install/dir/bin")
```

Save this file as eas-tools.lua, typically in the EAR/installation/path/etc/module, and load it with the command `module load eas-tools`.
[^2]: `<prefix>` is the location where you have installed the tool, e.g., the virtual environment installation directory, the value of `$PYTHONUSERBASE` environment variable in the case you use it.

## Usage

You must choose one of the three main required options.
The one you may use most of times is [`--format`](#--format), but the order followed in this document is useful for new users to understand how the tool works.

### `--print-config`

Pretty prints the [configuration](Configuration) being used by the tool.
You can take the printed configuration as an example for making yours and use it later through `--config-file` option.
The usage of this flag is very simple:

```bash
ear-job-visualizer --print-config > my_config.json
```

### `--avail-metrics`

Shows metrics names supported by tool.
These supported metrics are taken from the configuration file, so you can view the default supported metrics with:

```bash
ear-job-visualizer --avail-metrics
```

You can also check your own configuration file:

```bash
ear-job-visualizer --avail-metrics -c my_config.json
```

### `--format`

This option is in fact used to request for plotting (or converting) data.

Choices for this option are either [`runtime`](#runtime-format) or [`ear2prv`](#ear2prv-format), and each one enables each of the tool's features.
Read below sections for a detailed description of each one.

The **`runtime`** option is the one used to generate **static images**, while **`ear2prv`** refers the tool's interface to **output data following the Paraver Trace Format**.
Both format options share a subset of arguments.

The **`--job-id`** flag is **mandatory** to be specified.
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

Mentioned `--loops-file` and `--apps-file` options accept also a path to a directory instead of a filename.
This is useful because when you request EAR to generate csv files through the `--ear-user-db=<csv-filename>` flag, one csv file for each compute node is created.
Therefore, for each compute node your application ran on, files `<csv-filename>_<nodename>_loops.csv` and `<csv-filename>_<nodename>_apps.csv` are created.
Consequently, if you want to visualize runtime metrics for your specific multi-node application, you may need to move all *loops* and *apps* data into a single directory, respectively, and pass such directories to the tool's `loops-file` and `apps-file.`

```bash
mkdir apps_dir && mv *_apps.csv apps_dir

mkdir loops_dir && mv *_loops.csv loops_dir

ear-job-visualizer --format <format-option> --job-id <job-id> --loops-file loops_dir --apps_file apps_dir <format-specific-options>
```

### *runtime* format

Generate a heatmap-based graph for each metric specified by `--metrics` argument (i.e., space separated list of metric names).
Note that the accepted metrics are specified in the [configuration](Configuration) file and you can request the list trough the `--avail-metrics` flag.

> This option just supports plotting data for a single Job-Step ID, thus **both `--job-id` and `--step-id` flags are required**.

The resulting figure (for each requested metric) will be a timeline where for each node your application had used you will see a heatmap showing an intuitive visualisation about the value of the metric during application execution.
All nodes visualised share the same timeline, which makes this command useful to check the application behaviour over all of them.
Below there is an example showing how to generate images for a two-node MPI application, the I/O rate, the GFLOPS and the percentage of time spent in MPI calls along the execution time.

```bash
ear-job-visualizer --format runtime --job-id <> --step-id <> -m io_mbs gflops perc_mpi
```

> Use [`--avail-metrics`](#--avail-metrics) flag to view tool's supported metrics and the name you must use to retrieve them.

The above command line generates the following figures:

![An example of OpenRadioss GFLOPS across the execution time.]()

![An example of OpenRadioss I/O rate across the execution time.]()

![An example of OpenRadioss %MPI rate across the execution time.]()

#### GPU data

If you request GPU metrics, the graph will show you per-GPU data.
For each requested GPU metric the tool filters those GPUs which have a constant zero value along the execution time.

```bash
ear-job-visualizer --format runtime --job-id 69478 --step-id 0 --loops-file /examples/runtime_format/69478_loops.csv --apps-file /examples/runtime_format/69478_apps.csv -m gpu_util gpu_power -o 69478.0.png
```

The above command line generates the following figures:

![An example of GPU utilization of a single node application using just one GPU device.](/examples/runtime_format/runtime_gpu_util-69478.0.png)

![An example of GPU power consumption of a single node application using just one GPU device.](/examples/runtime_format/runtime_gpu_power-69478.0.png)

You can use the EAR `dcgmi.so` [report plug-in](https://gitlab.bsc.es/ear_team/ear/-/wikis/User-guide#runtime-report-plug-ins) to generate CSV files containing extra GPU metrics taken from either[^3]:

- The NVIDIA® Data Center GPU Manager ([DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html#profiling-metrics)).
- NVIDIA Management Library ([NVML](https://developer.nvidia.com/management-library-nvml)) [GPM metrics](https://docs.nvidia.com/deploy/nvml-api/group__nvmlGpmEnums.html#group__nvmlGpmEnums).

You can use later those csv files directly by invoking the tool with both `--loops-file` and `--apps-file` flags as well.

[^3]: The source of these metrics is transparent from the user point of view. In fact is EAR who takes data from the available source. Metrics are the same regardless the interface used.

#### Data visualization colormap range

By default, the colormap of the data is computed from the data value range found in the source, i.e., a colormap is built taken the minimum and maximum values of the requested metric along the runtime across all involed nodes/GPUs.
However, you can change this behaviour by passing the `--manual-range` flag. Thus, the tool will use the range for the requested metric specified at the [Configuration](#Configuration) file.

### *ear2prv* format

Convert job runtime data gathered from EARL to Paraver Trace Format.
In this case, all metrics found in the input data are reported to the trace file.
Moreover, you can have in the same trace all steps and applications (e.g., a workflow) of your job, so just the `--job-id` flag is required.

Keep in mind that the trace file generated by this tool have the following mapping between EAR data and the [Paraver Trace Format](https://tools.bsc.es/doc/1370.pdf):
- As EAR data is reported at the node-level, EAR node data can be visualized at the Paraver task-level (Thread 1 is used).
- The tool uses the thread-level to put the GPU data.

You can find two examples of [Paraver Config Files](examples) to easily start working with the output data generated by this option.

## Configuration

Check the [config.json](src/ear_job_visualize/config.json) file.

## Contact

For any question and suggestion, contact with support@eas4dc.com.
You can also open an issue in this repository.
