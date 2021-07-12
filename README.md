# ear-analytics

A tool to automatically read and visualize results provided by files get by 
_eacct_ command from [EAR](https://gitlab.bsc.es/ear_team/ear/-/wikis/home) framework.
**ear-analytics** is a cli program written in Python which lets you plot the data given by 
the output get when executing `eacct -j %slurm_job_id% -c %file_name%` or 
`eacct -j %slurm_job_id% -r -c %file_name%`. For more information, read about
[eacct](https://gitlab.bsc.es/ear_team/ear/-/wikis/Commands#energy-account-eacct) 
command.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.
You can install the requirements directly or use the requirements.txt file given with the source code.

### Requirements

Python 3.6.x

Output of `pip freeze` on a virtual environment used while deploying the 
application:
```
cycler==0.10.0
kiwisolver==1.3.1
matplotlib==3.3.4
numpy==1.19.5
pandas==1.1.5
Pillow==8.2.0
pyparsing==2.4.7
python-dateutil==2.8.1
pytz==2021.1
six==1.16.0
```

### Installation

`$ git clone`

```
$ python3 -m pip install -U pip
$ python3 -m pip install -r requirements.txt
```

#### Using a virtual environment

```
$ python3 -m venv env_name
$ source env_name/bin/activate
(env_name) $ python -m pip install -U pip
(env_name) $ python -m pip install -r requirements.txt

# If you want to quit from virtual env:
(env_name) $ deactivate
```

## Usage

If you are using a [virtual environment](#using-a-virtual-environment) remember to activate it.

```
$ python3 ear_analytics.py --help

usage: ear_analytics [-h] input_file {recursive,resume} ...

High level support for read and visualize information files given by eacct
command.

positional arguments:
  input_file          Specifies the input file name to read data from.

optional arguments:
  -h, --help          show this help message and exit

subcommands:
  Type `ear_analytics` <dummy_filename> {recursive,resume} -h to get more
  info of each subcommand

  {recursive,resume}  The two functionalities currently supported by this
                      program.
    recursive         Generate a heatmap graph showing the behaviour of some
                      metrics monitored with EARL. The file must be outputed
                      by `eacct -r` command.
    resume            Generate a resume about Energy and Power save, and Time
                      penalty of an application monitored with EARL.
```

You must provide the *input_file* name, which is a file in CSV format generated by **eacct** command. Then you must provide which sub-command you want to invoke. There are two options: [resume](#resume) and [recursive](#recursive).

### resume

```
$ python3 ear_analytics.py dummy resume --help

usage: ear_analytics input_file resume [-h] [--app_name APP_NAME] base_freq

positional arguments:
  base_freq            Specify which frequency is used as base reference for
                       computing and showing savings and penalties in the
                       figure.

optional arguments:
  -h, --help           show this help message and exit
  --app_name APP_NAME  Set the application name to get resume info.
```

Generate a bar plot that shows Energy and Power savings, and Time penalty of applying different policies to some application with respect to executions applying EAR's MONITORING policy.
You then must to specify at which nominal frequency *base_freq* the application was running on those reference MONITORING tests.

Note that if your *input_file* contains resume information of multiple apps, you can pass to this sub-command the option *--app_name APP_NAME*, which will let you filter the data of your input file and get results for only your desired application.

#### Example

The next table shows content of `examples/resume_multiple_apps.csv` file. We will visualize performance savings and penalties of `gromacs_4n_mt` application test with a **JOB-ID** 173277.
```
|    | JOB-STEP   | USER    | APPLICATION   | POLICY   |   NODES# |   FREQ(GHz) |   TIME(s) |   POWER(Watts) |      GBS |      CPI |   ENERGY(J) |   GFLOPS/WATT | G-POW (T/U)   | G-FREQ   | G-UTIL(G/MEM)   |   AVG IMC | DEF FREQ   |
|---:|:-----------|:--------|:--------------|:---------|---------:|------------:|----------:|---------------:|---------:|---------:|------------:|--------------:|:--------------|:---------|:----------------|----------:|:-----------|
|  0 | 173373-14  | xovidal | bqcd_4n       | MT       |        4 |     2.3184  |   138.218 |        264.022 | 10.3858  | 0.699178 |      145971 |      0.336029 | ---           | ---      | ---             |      1.99 | 2.1        |
|  1 | 173373-13  | xovidal | bqcd_4n       | ME       |        4 |     2.38178 |   134.215 |        278.651 | 10.6786  | 0.696412 |      149597 |      0.327248 | ---           | ---      | ---             |      2.13 | 2.4        |
|  2 | 173373-12  | xovidal | bqcd_4n       | MT       |        4 |     2.32575 |   138.222 |        264.781 | 10.3787  | 0.707558 |      146394 |      0.334691 | ---           | ---      | ---             |      1.99 | 2.1        |
|  3 | 173373-11  | xovidal | bqcd_4n       | ME       |        4 |     2.38035 |   132.205 |        285.401 | 10.841   | 0.688572 |      150926 |      0.324666 | ---           | ---      | ---             |      2.24 | 2.4        |
|  4 | 173373-10  | xovidal | bqcd_4n       | MT       |        4 |     2.3202  |   137.242 |        268.597 | 10.4839  | 0.694782 |      147451 |      0.332469 | ---           | ---      | ---             |      2.06 | 2.1        |
|  5 | 173373-9   | xovidal | bqcd_4n       | ME       |        4 |     2.37968 |   133.212 |        282.621 | 10.7625  | 0.686519 |      150594 |      0.324704 | ---           | ---      | ---             |      2.18 | 2.4        |
|  6 | 173373-8   | xovidal | bqcd_4n       | MO       |        4 |     2.37263 |   131.206 |        297.9   | 10.9351  | 0.689353 |      156345 |      0.313245 | ---           | ---      | ---             |      2.39 | 2.4        |
|  7 | 173373-7   | xovidal | bqcd_4n       | MO       |        4 |     2.37398 |   130.204 |        297.983 | 11.0039  | 0.66511  |      155195 |      0.315298 | ---           | ---      | ---             |      2.39 | 2.4        |
|  8 | 173373-6   | xovidal | bqcd_4n       | MO       |        4 |     2.3727  |   131.213 |        297.437 | 10.9146  | 0.66051  |      156111 |      0.313516 | ---           | ---      | ---             |      2.39 | 2.4        |
|  9 | 173373-5   | xovidal | bqcd_4n       | MO       |        4 |     2.08785 |   148.242 |        241.716 |  9.69047 | 0.675449 |      143330 |      0.340969 | ---           | ---      | ---             |      1.98 | 2.1        |
| 10 | 173373-4   | xovidal | bqcd_4n       | MO       |        4 |     2.088   |   148.238 |        241.071 |  9.70067 | 0.684292 |      142943 |      0.343012 | ---           | ---      | ---             |      1.98 | 2.1        |
| 11 | 173373-3   | xovidal | bqcd_4n       | MO       |        4 |     2.08792 |   148.23  |        242.956 |  9.69146 | 0.678862 |      144053 |      0.340109 | ---           | ---      | ---             |      1.98 | 2.1        |
| 12 | 173373-2   | xovidal | bqcd_4n       | MO       |        4 |     2.6109  |   126.181 |        322.89  | 11.3331  | 0.6771   |      162970 |      0.300268 | ---           | ---      | ---             |      2.39 | TURBO      |
| 13 | 173373-1   | xovidal | bqcd_4n       | MO       |        4 |     2.61817 |   126.199 |        322.553 | 11.3207  | 0.712089 |      162824 |      0.300907 | ---           | ---      | ---             |      2.39 | TURBO      |
| 14 | 173373-0   | xovidal | bqcd_4n       | MO       |        4 |     2.60925 |   126.227 |        321.029 | 11.3235  | 0.689998 |      162090 |      0.302251 | ---           | ---      | ---             |      2.39 | TURBO      |
| 15 | 173277-11  | xovidal | gromacs_4n_mt | ME       |        4 |     2.27272 |   314.221 |        299.01  | 10.4793  | 0.469019 |      375821 |      1.69575  | ---           | ---      | ---             |      1.99 | 2.4        |
| 16 | 173277-10  | xovidal | gromacs_4n_mt | MT       |        4 |     2.1684  |   318.264 |        294.925 | 10.4113  | 0.427339 |      375455 |      0.312342 | ---           | ---      | ---             |      1.99 | 2.1        |
| 17 | 173277-9   | xovidal | gromacs_4n_mt | ME       |        4 |     2.27422 |   314.214 |        297.563 | 10.432   | 0.479014 |      373995 |      1.98823  | ---           | ---      | ---             |      1.96 | 2.4        |
| 18 | 173277-8   | xovidal | gromacs_4n_mt | MT       |        4 |     2.16878 |   317.279 |        294.772 | 10.2703  | 0.498848 |      374100 |      2.41074  | ---           | ---      | ---             |      1.99 | 2.1        |
| 19 | 173277-7   | xovidal | gromacs_4n_mt | ME       |        4 |     2.27438 |   313.216 |        297.905 | 10.6963  | 0.463183 |      373235 |      1.66359  | ---           | ---      | ---             |      1.96 | 2.4        |
| 20 | 173277-6   | xovidal | gromacs_4n_mt | MT       |        4 |     2.16863 |   319.292 |        294.503 | 10.2747  | 0.531323 |      376130 |      3.37354  | ---           | ---      | ---             |      1.99 | 2.1        |
| 21 | 173277-5   | xovidal | gromacs_4n_mt | MO       |        4 |     2.2773  |   313.213 |        323.426 | 10.4221  | 0.495897 |      405205 |      2.41883  | ---           | ---      | ---             |      2.39 | 2.4        |
| 22 | 173277-4   | xovidal | gromacs_4n_mt | MO       |        4 |     2.27888 |   311.222 |        323.349 | 10.4382  | 0.525289 |      402533 |      3.17632  | ---           | ---      | ---             |      2.39 | 2.4        |
| 23 | 173277-3   | xovidal | gromacs_4n_mt | MO       |        4 |     2.088   |   334.255 |        278.723 |  9.77158 | 0.531219 |      372657 |      3.40118  | ---           | ---      | ---             |      1.97 | 2.1        |
| 24 | 173277-2   | xovidal | gromacs_4n_mt | MO       |        4 |     2.088   |   336.286 |        278.573 |  9.74747 | 0.4831   |      374721 |      1.88377  | ---           | ---      | ---             |      1.97 | 2.1        |
| 25 | 173277-1   | xovidal | gromacs_4n_mt | MO       |        4 |     2.54108 |   307.198 |        341.611 | 10.7149  | 0.476542 |      419769 |      1.77675  | ---           | ---      | ---             |      2.39 | TURBO      |
| 26 | 173277-0   | xovidal | gromacs_4n_mt | MO       |        4 |     2.54257 |   309.182 |        341.91  | 10.5852  | 0.511182 |      422851 |      2.16451  | ---           | ---      | ---             |      2.39 | TURBO      |
```

We can type:

`$ python3 ear_analytics.py examples/resume_multiple_apps.csv resume --app_name gromacs_4n_mt`

Therefore we obtain:

![alt text](resume_mult_apps.png)
