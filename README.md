# ear-analytics

A tool to automatically read and visualize results provided by files get by 
_eacct_ command from [EAR](https://gitlab.bsc.es/ear_team/ear/-/wikis/home) framework.
**ear-analytics** is a cli program written in Python which lets you plot the data given by 
the output get when executing `eacct -j %slurm_job_id% -c %file_name%` or 
`eacct -j %slurm_job_id% -l -c %file_name%`. For more information, read about
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
