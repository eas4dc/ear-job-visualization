# ear-job-analytics

A tool to automatically read and visualise data provided by the [EAR](https://gitlab.bsc.es/ear_team/ear/-/wikis/home) framework.
**ear-analytics** is a cli program written in Python which lets you plot the EAR data given by some of its commands or by using some report plug-in offered by the EAR Library (EARL).
The main visualisation target is to show runtime metrics collected by EARL in a timeline graph.
By now this tool supports two kind of output formats: (I) directly generate images showing runtime information, (II) generate a trace file to be read by Paraver, a tool to visualise and manage trace data maintaned by the Barcelona Supercomputing Center's Tools team.

For more information, read about [eacct](https://gitlab.bsc.es/ear_team/ear/-/wikis/EAR-commands#ear-job-accounting-eacct) or [this guide](https://gitlab.bsc.es/ear_team/ear/-/wikis/User%20guide#running-jobs-with-ear) which shows you how to run jobs with EAR and how to obtain runtime data.
You can find [here](https://tools.bsc.es/paraver) more information about how Paraver works.

## Features

- Generate static images showing runtime metrics of your job monitored by EARL.
    - Figures can be displayed.
- Generate Paraver traces to visualize runtime metrics within Paraver tool or any other tool of the BSC's Tools teams.
- **(New)** Generate a LaTeX project with the most relevant information about the job to be analyzed.
    - Job global summary.
    - Job phase classification.
    - Job runtime metrics.

## Requirements

- Python < 3.9
- Numpy
- Matplotlib < 3.5
- Proplot
- Pandas

By default, the tool calls internally the EAR account command (i.e., *eacct*) with the proper information and options in order to get the corresponding data to be sent to the tool's functionalities.
Be sure you have the the *eacct* command on your path, and also check whether `EAR_ETC` environment variable is set properly.

If you have some trouble, ask your system administrator if there is some problem with the EAR Database.
You can also provide directly input files if eacct is unable, [read below](providing-files-instead-of-using-internally-eacct). 

## Installation

This repository contains all recipes to build and install the package.
You need **build** and **setuptools** packages properly build and install this one.

```
pip install -U pip
pip install build setuptools
python -m build
pip install .
```

Then, you can type `ear-job-analytics` and you should see the following:

```
usage: ear-job-analytics [-h] [--version] --format {runtime,ear2prv,summary}
                         [--input-file INPUT_FILE] -j JOB_ID -s STEP_ID
                         [--save | --show] [-t TITLE] [-r] [-l]
                         [-m metric [metric ...]] [-e]
                         [--events-config EVENTS_CONFIG] [-o OUTPUT] [-k]
ear-job-analytics: error: the following arguments are required: --format, -j/--job-id, -s/--step-id
```

If you had some trouble during the build and/or installation process, contact with oriol.vidal@eas4dc.com.
We are trying provide a more easy way to install the package.

## Usage

It is mandatory to specify the output format (i.e., `--format`) you want produce.
Choices for this option are either *runtime*, *ear2prv* or *job-summary*, and each one enables each of the tool's features.
Read below for a detailed description of each feature.

In addition, you must specify the Job (i.e., `--job-id`) and Step (i.e., `--step-id`) IDs of the job being analyzed as features currently only support working with data corresponding with one Job-Step.
These required options ensures the tool to filter the input data by Job and Step IDs, respectively, to avoid possible errors on the output.
So, the minimum shape of an invokation is:

```
$> ear-job-analytics --format [runtime|ear2prv|summary] --job-id <JobID> --step-id <StepID>
```

The *runtime* option is the one used to generate static images (which you can modify at invokation time), while *ear2prv* refers the tool's interface to output data following the Paraver Trace Format.
Finally, *job-summary* generates an overview analysis of the most relevant information of the job.

** _ear2prv_ format is not in production yet. **

### Providing files instead of using internally eacct

If you know which *eacct* invokations are required to visualise the data, you can use the option *--input-file* to specify where the tool will find the data to be filtered by the two required job-related options (e.g., *--job-id*, *--step-id*).
This option is useful when you already have data for multiple jobs and/or steps together and you want to work on it in several ways because naturally it's more fast to work directly on a file than invoking a command to make a query to a Database, storing the output on a file, and then read such file.
This option is also useful since it lets you work on a host where you can't access EAR Database nor EAR is installed.

The way how the value of this option is handled depends on which functionality (e.g., *format*) you are working on, and which kind of data you want to produce/visualise.
If **runtime** format option is used, the *--input-file* option can be a single filename (which can be given with its relative path) wich contains EAR loop data.
If a directory name is given, the tool will read all files inside it (another reason why it is required to specify the Job and Step IDs).

If you started working by using *eacct* command internally, all required files are stored temporally while the tool is doing its work.
If you want to reuse such files later you can pass the option `--keep-csv` to prevent files been removed.
Then, you can provide those files to get different output.

### *runtime* format

Generate a heatmap-based graph for each metric specified by `--metrics` argument (i.e., space separated list of metric names).
Note that the accepted metrics by your **ear-analytics** installation are specified in the configuration file.

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

## Contact

For any question and suggestion, contact with support@eas4dc.com.
You can also open an issue in this repository.
